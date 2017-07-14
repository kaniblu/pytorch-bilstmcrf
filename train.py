import os
import random
import pickle
import shutil
import tempfile
import datetime
import subprocess
import multiprocessing.pool as mp

import numpy as np
import tqdm
import torch
import torch.optim as O
from torch.nn.utils import clip_grad_norm
from torch.autograd import Variable

from utils.argparser import ArgParser
from utils.argparser import path
from utils.vocab import Vocabulary
from utils.generator import TextFileReader
from utils.generator import SentenceGenerator
from utils.preprocessor import Preprocessor
from utils.visdom import Visdom
from model import BiLSTMCRF
from evaluate import compute_f1
from evaluate import preprocess_labels


def parse_args():
    parser = ArgParser(allow_config=True)

    parser.add("--name", type=str, default="main")
    parser.add("--feats_path", type=path, action="append", required=True)
    parser.add("--feats_vocab", type=path, action="append", required=True)
    parser.add("--labels_path", type=path, required=True)
    parser.add("--labels_vocab", type=path, required=True)
    parser.add("--save_dir", type=path, required=True)
    parser.add("--gpu", action="store_true", default=False)
    parser.add("--n_previews", type=int, default=10)

    group = parser.add_group("Word Embedding Options")
    group.add("--wordembed_type", type=str, action="append",
              choices=["glove", "fasttext", "none"])
    group.add("--wordembed_path", type=path, action="append")
    group.add("--fasttext_path", type=path, default=None)
    group.add("--wordembed_freeze", type=bool, action="append")

    group = parser.add_group("Training Options")
    group.add("--n_epochs", type=int, default=3)
    group.add("--dropout_prob", type=float, default=0.05)
    group.add("--batch_size", type=int, default=32)
    group.add("--max_len", type=int, default=30)

    group = parser.add_group("Save Options")
    group.add("--save", action="store_true", default=False)
    group.add("--save_period", type=int, default=1000)

    group = parser.add_group("Validation Options")
    group.add("--val", action="store_true", default=False)
    group.add("--val_period", type=int, default=100)
    group.add("--text_preview", action="store_true", default=False)
    group.add("--val_feats_path", type=path, action="append")
    group.add("--val_labels_path", type=path, default=None)

    group = parser.add_group("Visdom Options")
    group.add("--visdom_host", type=str, default="localhost")
    group.add("--visdom_port", type=int, default=8097)
    group.add("--visdom_buffer_size", type=int, default=10)

    group = parser.add_group("Model Parameters")
    group.add("--word_dim", type=int, action="append")
    group.add("--hidden_dim", type=int, required=True)

    args = parser.parse_args()

    return args


def load_glove_embeddings(embeddings, vocab, path):
    word_dim = embeddings.embedding_dim
    with open(path, 'r') as f:
        for i, line in enumerate(f):
            tokens = line.split()
            word = " ".join(tokens[:-word_dim])

            if word not in vocab.f2i:
                continue

            idx = vocab.f2i[word]
            values = [float(v) for v in tokens[-word_dim:]]
            embeddings.weight.data[idx] = (torch.FloatTensor(values))


def load_fasttext_embeddings(embeddings, vocab, fasttext_path, embedding_path):
    fasttext_path = os.path.abspath(fasttext_path)
    temp_dir = tempfile.gettempdir()
    query_path = os.path.join(temp_dir, "queries.txt")
    out_path = os.path.join(temp_dir, "vecs.txt")

    with open(query_path, "w") as f:
        f.write("\n".join(vocab.words))

    p1 = subprocess.Popen(["cat", query_path], stdout=subprocess.PIPE)
    output = subprocess.check_output([fasttext_path,
                                      "print-word-vectors",
                                      embedding_path],
                                     stdin=p1.stdout)
    with open(out_path, "w") as f:
        f.write(output.decode("utf-8"))

    load_glove_embeddings(embeddings, vocab, out_path)


def prepare_batch(xs, y, lens, gpu=False, volatile=False):
    lens, idx = torch.sort(lens, 0, True)
    xs, y = [x[idx] for x in xs], y[idx]
    xs = torch.cat([x.unsqueeze(0) for x in xs])

    xs = Variable(xs, volatile=volatile)
    y = Variable(y, volatile=volatile)
    lens = Variable(lens, volatile=volatile)

    if gpu:
        xs = xs.cuda(async=True)
        y = y.cuda(async=True)
        lens = lens.cuda(async=True)

    return xs, y, lens


def val_texts(xs_sents, y_sents, lstm_sents, crf_sents):
    texts = []

    for j, (xs_sent, y_sent, lstm_sent, crf_sent) in \
            enumerate(zip(xs_sents, y_sents, lstm_sents, crf_sents)):
        text = ""

        for i, x_sent in enumerate(xs_sent):
            text += "Feature_{}: {}\n".format(i, x_sent)

        text += "Target: {}\n".format(y_sent)
        text += "BiLSTM Output: {}\n".format(lstm_sent)
        text += "CRF Output: {}\n".format(crf_sent)
        texts.append(text)

    return texts


def to_sent(vocab, seq):
    return " ".join(vocab.i2f[x] if x in vocab.i2f else vocab.unk
                    for x in seq)


def val_sents(feat_vocabs, label_vocab, xs, y, lstm_preds, crf_preds, lens):
    xs = np.transpose(xs, (1, 0, 2))
    xs_sents = [[to_sent(feat_vocab, x_[:l])
                 for feat_vocab, x_ in zip(feat_vocabs, x)]
                for x, l in zip(xs, lens)]
    y_sents = [to_sent(label_vocab, y_[:l])
               for y_, l in zip(y, lens)]
    lstm_sents = [to_sent(label_vocab, x[:l])
                  for x, l in zip(lstm_preds, lens)]
    crf_sents = [to_sent(label_vocab, x[:l])
                 for x, l in zip(crf_preds, lens)]

    return xs_sents, y_sents, lstm_sents, crf_sents


def prepare_val_texts(model, batch_xs, batch_y, batch_lens,
                     logits, preds, n_previews):
    idx = np.random.permutation(np.arange(batch_lens.size(0)))[:n_previews]
    idx_v = Variable(torch.LongTensor(idx), volatile=True)

    if model.is_cuda:
        idx_v = idx_v.cuda()

    logits = torch.index_select(logits, 0, idx_v)
    bilstm_preds = logits.cpu().max(2)[1].squeeze(-1).data.numpy()
    crf_preds = preds.cpu().data.numpy()[idx]
    xs = batch_xs.cpu().data.numpy()[:, idx]
    y = batch_y.cpu().data.numpy()[idx]
    lens = batch_lens.cpu().data.numpy()[idx]

    sents = val_sents(model.word_vocabs, model.label_vocab,
                      xs, y, bilstm_preds, crf_preds, lens)
    texts = val_texts(*sents)

    return texts


def validate(model, data_loader_fn, viz_pool, preview_enabled, n_previews,
             train_instances):
    legend = ["Loss", "Precision", "Accuracy", "F1-Score"]
    title = "Validation"
    t = tqdm.tqdm()

    nll_all = 0.0
    n_instances = 0
    loaders = data_loader_fn()
    texts_all = []
    preds_all = []
    y_all = []

    for data in zip(*loaders):
        xs = [x[0] for x in data[:-1]]
        y, lens = data[-1]
        batch_size = len(lens)

        batch_xs, batch_y, batch_lens = prepare_batch(xs, y, lens,
                                                      gpu=model.is_cuda,
                                                      volatile=True)
        loglik, logits = model.loglik(batch_xs, batch_y, batch_lens)
        scores, preds = model.crf.viterbi_decode(logits, batch_lens)
        negloglik = -loglik.mean()
        negloglik_v = float(negloglik.data[0])

        if preview_enabled:
            texts = prepare_val_texts(model, batch_xs, batch_y, batch_lens,
                                    logits, preds, n_previews)
            texts_all.extend(texts)

        preds = preds.cpu().data.tolist()
        y = batch_y.cpu().data.tolist()
        lens = batch_lens.cpu().data.tolist()

        def _to_taglist(vocab, x):
            return [[vocab.i2f[w] if w in vocab.i2f else vocab.unk
                     for w in sent] for sent in x]

        preds = [pred[:l] for pred, l in zip(preds, lens)]
        y = [_y[:l] for _y, l in zip(y, lens)]
        preds = _to_taglist(model.label_vocab, preds)
        y = _to_taglist(model.label_vocab, y)

        preds = preprocess_labels(preds)
        y = preprocess_labels(y)

        preds_all.extend(preds)
        y_all.extend(y)
        nll_all += negloglik_v * batch_size

        t.set_description(
            "[{}]: validation loss={:.4f}".format(n_instances, negloglik_v))
        t.update(batch_size)
        n_instances += batch_size

    nll_all /= n_instances
    prec, rec, f1 = compute_f1(preds_all, y_all)

    if preview_enabled:
        texts = random.sample(texts_all, n_previews)
        viz_run("code", tuple(), dict(
            text="Instance {}\n".format(train_instances) + "\n".join(texts),
            opts=dict(
                title="Validation Text"
            )
        ))

    plot_X = [train_instances] * 4
    plot_Y = [nll_all, prec, rec, f1]

    viz_run("plot", tuple(), dict(
        X=[plot_X],
        Y=[plot_Y],
        opts=dict(
            legend=legend,
            title=title
        ),
        flush=True
    ))


def train(model, data_loader_fn, val_data_loader_fn, n_epochs, viz_pool,
          val_period, n_previews, val_enabled, preview_enabled,
          save_enabled, save_dir, save_period):
    params = set(p for p in model.parameters() if p.requires_grad)
    optimizer = O.Adam(params)
    legend = ["-Log Likelihood"]
    step = 0
    n_instances = 0
    t = tqdm.tqdm()

    for _ in range(n_epochs):
        loaders = data_loader_fn()

        for data in zip(*loaders):
            model.zero_grad()

            xs = [x[0] for x in data[:-1]]
            y, lens = data[-1]
            batch_size = len(lens)
            n_instances += batch_size
            step += 1

            batch_xs, batch_y, batch_lens = prepare_batch(xs, y, lens,
                                                          gpu=model.is_cuda,
                                                          volatile=False)
            loglik, logits = model.loglik(batch_xs, batch_y, batch_lens)
            negloglik = -loglik.mean()
            negloglik_v = float(negloglik.data[0])

            plot_X = n_instances
            plot_Y = negloglik_v

            negloglik.backward()
            clip_grad_norm(model.parameters(), 3)
            optimizer.step()

            viz_run("plot", tuple(), dict(
                X=[plot_X],
                Y=[plot_Y],
                opts=dict(
                    legend=legend,
                    title="Training Loss"
                )
            ))

            if val_enabled and step % val_period == 0:
                validate(model, val_data_loader_fn, viz_pool, preview_enabled,
                         n_previews, n_instances)

            if save_enabled and step % save_period == 0:
                model_filename = "model-{}-{:.4f}".format(
                    n_instances, abs(negloglik_v))
                path = os.path.join(save_dir, model_filename)
                torch.save(model.state_dict(), path)
                viz.save([save_dir])

            t.set_description(
                "[{}]: loss={:.4f}".format(n_instances, negloglik_v))
            t.update(batch_size)


def init_viz(args, kwargs):
    global viz

    viz = Visdom(*args, **kwargs)


def viz_run(f_name, args=(), kwargs=dict()):
    global viz

    getattr(viz, f_name).__call__(*args, **kwargs)


def _load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    args = parse_args()

    print("Loading vocabulary...")
    feats_vocabs = [_load_vocab(path) for path in args.feats_vocab]
    labels_vocab = _load_vocab(args.labels_vocab)

    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_basename = timestamp + "-{}".format(args.name)
    save_dir = os.path.join(args.save_dir, save_basename)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir, exist_ok=True)

    print("Initializing model...")
    model_cls = BiLSTMCRF
    model = model_cls(feats_vocabs, labels_vocab, args.word_dim,
                      args.hidden_dim,
                      dropout_prob=args.dropout_prob)

    model.reset_parameters()

    print("Loading word embeddings...")

    for i, (vocab, we_type, we_path, we_freeze) in \
            enumerate(
                zip(feats_vocabs, args.wordembed_type, args.wordembed_path,
                    args.wordembed_freeze)):
        embeddings = getattr(model, "embeddings_{}".format(i))

        if we_type == "glove":
            load_glove_embeddings(embeddings, vocab, we_path)
        elif we_type == "fasttext":
            load_fasttext_embeddings(embeddings, vocab,
                                     fasttext_path=args.fasttext_path,
                                     embedding_path=we_path)
        elif we_type == "none":
            pass
        else:
            raise ValueError("Unrecognized word embedding type: {}".format(
                we_type
            ))

        if we_freeze:
            embeddings.weight.requires_grad = False

    if args.gpu:
        model = model.cuda()

    viz_pool = mp.ThreadPool(1, initializer=init_viz, initargs=(tuple(), dict(
        buffer_size=args.visdom_buffer_size,
        server="http://{}".format(args.visdom_host),
        port=args.visdom_port,
        env=args.name,
        name=timestamp
    )))

    viz_pool.apply_async(viz_run, ("code", tuple(), dict(
        text=str(args)[10:-1].replace(", ", "\n"),
        opts=dict(
            title="Arguments"
        )
    )))

    config_path = os.path.join(save_dir, os.path.basename(args.config))
    shutil.copy(args.config, config_path)

    def _data_loader_fn_generator(feats_vocabs, labels_vocab, feats_paths,
                                  labels_path):
        def _data_loader_fn():
            feats_preps = [Preprocessor(vocab, add_bos=False, add_eos=False)
                           for vocab in feats_vocabs]
            labels_prep = Preprocessor(labels_vocab,
                                       add_bos=False, add_eos=False)
            feats_readers = [TextFileReader(path) for path in feats_paths]
            labels_reader = TextFileReader(labels_path)

            feats_gen = [SentenceGenerator(reader, vocab, args.batch_size,
                                           max_length=args.max_len,
                                           preprocessor=prep,
                                           allow_residual=True)
                         for reader, vocab, prep in
                         zip(feats_readers, feats_vocabs, feats_preps)]
            labels_gen = SentenceGenerator(labels_reader, labels_vocab,
                                           args.batch_size,
                                           max_length=args.max_len,
                                           preprocessor=labels_prep,
                                           allow_residual=True,)

            return feats_gen + [labels_gen]

        return _data_loader_fn

    _data_loader_fn = _data_loader_fn_generator(feats_vocabs=feats_vocabs,
                                                labels_vocab=labels_vocab,
                                                feats_paths=args.feats_path,
                                                labels_path=args.labels_path)
    _val_data_loader_fn = _data_loader_fn_generator(feats_vocabs=feats_vocabs,
                                                    labels_vocab=labels_vocab,
                                                    feats_paths=args.val_feats_path,
                                                    labels_path=args.val_labels_path)

    print("Beginning training...")
    train(model,
          data_loader_fn=_data_loader_fn,
          val_data_loader_fn=_val_data_loader_fn,
          n_epochs=args.n_epochs,
          viz_pool=viz_pool,
          save_dir=save_dir,
          n_previews=args.n_previews,
          save_period=args.save_period,
          val_enabled=args.val,
          val_period=args.val_period,
          preview_enabled=args.text_preview,
          save_enabled=args.save)

    # Flush remaining buffer
    viz_run("flush", tuple(), dict())

    print("Done!")


if __name__ == '__main__':
    main()
