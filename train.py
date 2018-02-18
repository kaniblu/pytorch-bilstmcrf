import os
import pickle
import shutil
import logging
import argparse
import tempfile
import subprocess
import collections

import numpy as np
import yaap
import tqdm
import torch
import torch.nn as nn
import torch.optim as O
import torch.autograd as A

import utils
import data as D
import model as M
import evaluate as E


parser = yaap.ArgParser(
    allow_config=True,
    formatter_class=argparse.ArgumentDefaultsHelpFormatter
)

group = parser.add_group("Basic Options")
group.add("--input-path", type=yaap.path, action="append", required=True,
          help="Path to input file that contains sequences of tokens "
               "separated by spaces.")
group.add("--label-path", type=yaap.path, required=True,
          help="Path to label file that contains sequences of token "
               "labels separated by spaces. Note that the number of "
               "tokens in each sequence must be equal to that of the "
               "corresponding input sequence.")
group.add("--save-dir", type=yaap.path, required=True,
          help="Directory to save outputs (checkpoints, vocabs, etc.)")
group.add("--gpu", type=int, action="append",
          help="Device id of gpu to use. Could supply multiple gpu ids "
               "to denote multi-gpu utilization. If no gpus are "
               "specified, cpu is used as default.")
group.add("--tensorboard", action="store_true", default=False,
          help="Whether to enable tensorboard visualization. Requires "
               "standalone tensorboard, which can be installed via "
               "'https://github.com/dmlc/tensorboard'.")

group = parser.add_group("Word Embedding Options")
group.add("--wordembed-type", type=str, action="append",
          choices=["glove", "fasttext", "none"],
          help="Type of pretrained word embeddings to use for each input. "
               "If multiple input paths are supplied, the same number of "
               "this option must be specified as well. If no option is "
               "supplied, no word embeddings will be used.")
group.add("--wordembed-path", type=yaap.path, action="append",
          help="Path to pre-trained word embeddings. "
               "If embedding type is 'glove', glove-style embedding "
               "file is expected. If embedding type is 'fasttext', "
               "fasttext model file is expected. The number of "
               "specifications must match the number of inputs.")
group.add("--fasttext_path", type=yaap.path, default=None,
          help="If embedding type is 'fasttext', path to fasttext "
               "binaries must be specified. Otherwise, this option is "
               "ignored.")
group.add("--wordembed-freeze", type=bool, action="append",
          help="Whether to freeze embeddings matrix during training. "
               "The number of specifications must match the number of "
               "inputs. If none is specified, word embeddings will not be "
               "frozen by default.")

group = parser.add_group("Training Options")
group.add("--epochs", type=int, default=3,
          help="Number of training epochs.")
group.add("--dropout-prob", type=float, default=0.05,
          help="Probability in dropout layers.")
group.add("--batch-size", type=int, default=32,
          help="Mini-batch size.")
group.add("--shuffle", action="store_true", default=False,
          help="Whether to shuffle the dataset.")
# group.add("--max-len", type=int, default=30,
#           help="Maximum length of sequences. If a training example "
#                "is longer than this option, it will be skipped.")
group.add("--ckpt-period", type=utils.PeriodChecker, default="1e",
          help="Period to wait until a model checkpoint is "
               "saved to the disk. "
               "Periods are specified by an integer and a unit ('e': "
               "epoch, 'i': iteration, 's': global step).")

group = parser.add_group("Validation Options")
group.add("--val", action="store_true", default=False,
          help="Whether to perform validation.")
group.add("--val-ratio", type=float, default=0.1)
group.add("--val-period", type=utils.PeriodChecker, default="100i",
          help="Period to wait until a validation is performed. "
               "Periods are specified by an integer and a unit ('e': "
               "epoch, 'i': iteration, 's': global step).")
group.add("--samples", type=int, default=10,
          help="Number of output samples to display at each iteration.")

group = parser.add_group("Model Parameters")
group.add("--word-dim", type=int, action="append",
          help="Dimensions of word embeddings. Must be specified for each "
               "input. Defaults to 300 if none is specified.")
group.add("--lstm-dim", type=int, default=300,
          help="Dimensions of lstm cells. This determines the hidden "
               "state and cell state sizes.")
group.add("--lstm-layers", type=int, default=1,
          help="Layers of lstm cells.")
group.add("--bidirectional", action="store_true", default=False,
          help="Whether lstm cells are bidirectional.")


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


class BaseLSTMCRFTrainer(object):
    def __init__(self, model: M.LSTMCRF, epochs=5, optimizer=O.Adam, gpus=None):
        self.model = model
        self.epochs = epochs
        self.optimizer_cls = optimizer
        self.optimizer = optimizer(self.model.parameters())
        self.gpus = gpus
        self.gpu_main = None

        if gpus is not None:
            self.gpus = self._ensure_tuple(self.gpus)
            self.gpu_main = self.gpus[0]
            self.model = M.TransparentDataParallel(
                self.model,
                device_ids=self.gpus,
                output_device=self.gpu_main
            )

    def wrap_var(self, x, **kwargs):
        x = A.Variable(x, **kwargs)

        if self.gpus is not None:
            x = x.cuda(self.gpu_main)

        return x

    @staticmethod
    def _ensure_tuple(x):
        if not isinstance(x, collections.Sequence):
            return (x, )

        return x

    def prepare_batch(self, xs, y, lens, **var_kwargs):
        lens, idx = torch.sort(lens, 0, True)
        xs, y = xs[:, idx], y[idx]
        xs, y = self.wrap_var(xs, **var_kwargs), self.wrap_var(y, **var_kwargs)
        lens = self.wrap_var(lens, **var_kwargs)

        return xs, y, lens

    def on_iter_complete(self, loss, local_iter, global_iter, global_step):
        pass

    def on_epoch_complete(self, epoch_idx, global_iter, global_step):
        pass

    def train(self, data, data_size=None):
        if data_size is not None:
            total_steps = self.epochs * data_size
        else:
            total_steps = None

        self.model.train(True)
        global_step = 0
        global_iter = 0
        progress = tqdm.tqdm(total=total_steps)

        for e_idx in range(self.epochs):
            e_idx += 1
            for i_idx, (batch, lens) in enumerate(data):
                batch_size = batch[0].size(0)
                i_idx += 1
                global_step += batch_size
                global_iter += 1
                self.model.zero_grad()

                xs, y = batch[:-1], batch[-1]
                xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens)

                loglik = self.model.loglik(xs_var, y_var, lens_s)
                nll = -loglik.mean()
                nll_v = float(-(loglik / lens_s.float()).data[0])
                nll.backward()
                self.optimizer.step()

                progress.update(batch_size)
                progress.set_description(f"nll={nll_v:.4f}")
                self.on_iter_complete(nll_v, i_idx, global_iter, global_step)
            self.on_epoch_complete(e_idx, global_iter, global_step)


class LSTMCRFTrainer(BaseLSTMCRFTrainer):
    def __init__(self, sargs, input_vocabs, label_vocab, *args,
                 val_data=None, **kwargs):
        super(LSTMCRFTrainer, self).__init__(*args, **kwargs)

        self.args = sargs
        self.input_vocabs = input_vocabs
        self.label_vocab = label_vocab
        self.val_data = val_data
        self.writer = None

        if self.args.tensorboard:
            self.writer = T.SummaryWriter(self.args.save_dir)

        self.repeatables = {
            self.args.ckpt_period: self.save_checkpoint
        }

        if self.args.val:
            self.repeatables[self.args.val_period] = \
                self.validate

    @staticmethod
    def get_longest_word(vocab):
        lens = [(len(w), w) for w in vocab.f2i]
        return max(lens, key=lambda x: x[0])[1]

    def display_row(self, items, widths):
        assert len(items) == len(widths)

        padded = ["{{:>{}s}}".format(w).format(item)
                  for item, w in zip(items, widths)]
        logging.info(" ".join(padded))

    def display_samples(self, inputs, targets, lstms, crfs):
        assert len(self.input_vocabs) == len(inputs), \
            "Number of input features do not match."

        inputs = [[self.lexicalize(s, v) for s in input]
                  for input, v in zip(inputs, self.input_vocabs)]
        targets = [self.lexicalize(s, self.label_vocab) for s in targets]
        lstms = [self.lexicalize(s, self.label_vocab) for s in lstms]
        crfs = [self.lexicalize(s, self.label_vocab) for s in crfs]
        transposed = list(zip(*(inputs + [lstms, crfs, targets])))
        col_names = [f"INPUT{i + 1:02d}" for i in range(len(inputs))] + \
            ["LSTM", "CRF", "TARGETS"]
        vocabs = self.input_vocabs + [self.label_vocab] * 3
        col_widths = [max(len(self.get_longest_word(v)), len(c))
                      for v, c in zip(vocabs, col_names)]

        for i, sample in enumerate(transposed):
            rows = list(zip(*sample))

            logging.info("")
            logging.info(f"SAMPLE #{i + 1}")
            self.display_row(col_names, col_widths)
            for row in rows:
                self.display_row(row, col_widths)

    @staticmethod
    def lexicalize(seq, vocab):
        return [vocab.i2f[w] if w in vocab else "<unk>" for w in seq]

    @staticmethod
    def tighten(seqs, lens):
        return [s[:l] for s, l in zip(seqs, lens)]

    @staticmethod
    def random_idx(max_count, subset=None):
        idx = np.random.permutation(np.arange(max_count))

        if subset is not None:
            return idx[:subset]

        return idx

    @staticmethod
    def gather(lst, idx):
        return [lst[i] for i in idx]

    def validate(self, epochs=None, iters=None, steps=None):
        if not self.args.val:
            return

        logging.info("Validating...")
        self.model.train(False)
        nll_all = 0
        preds_all, targets_all = [], []
        data_size = 0
        sampled = False

        for i_idx, (batch, lens) in enumerate(self.val_data):
            i_idx += 1
            batch_size = batch.size(1)
            data_size += batch_size

            xs, y = batch[:-1], batch[-1]
            xs_var, y_var, lens_s = self.prepare_batch(xs, y, lens,
                                                       volatile=True)
            loglik, logits = self.model.loglik(xs_var, y_var, lens_s,
                                               return_logits=True)
            nll = -loglik.mean()
            nll_v = float(-(nll / lens_s.float()).data[0])

            preds = self.model.predict(xs_var, lens_s)
            preds = preds.cpu().data.tolist()
            targets = y_var.cpu().data.tolist()
            lens_s = lens_s.cpu().data.tolist()

            preds = self.tighten(preds, lens_s)
            targets = self.tighten(targets, lens_s)

            preds_all.extend(preds)
            targets_all.extend(targets)
            nll_all = nll_v * batch_size

            if not sampled and self.args.samples > 0:
                sample_idx = self.random_idx(batch_size, self.args.samples)
                xs = xs_var.cpu().data.tolist()
                xs = [self.tighten(x, lens_s) for x in xs]
                lstm = logits.max(2)[1]
                lstm = lstm.cpu().data.tolist()
                lstm = self.tighten(lstm, lens_s)

                xs_smp = [self.gather(x, sample_idx) for x in xs]
                y_smp = self.gather(targets, sample_idx)
                crf_smp = self.gather(preds, sample_idx)
                lstm_smp = self.gather(lstm, sample_idx)

                self.display_samples(xs_smp, y_smp, lstm_smp, crf_smp)
                del xs, lstm, xs_smp, y_smp, crf_smp, lstm_smp

                sampled = True

        nll = nll_all / data_size
        preds_all = [self.lexicalize(s, self.label_vocab) for s in preds_all]
        targets_all = [self.lexicalize(s, self.label_vocab)
                       for s in targets_all]
        preds_all = E.preprocess_labels(preds_all)
        targets_all = E.preprocess_labels(targets_all)
        prec, rec, f1 = E.compute_f1(preds_all, targets_all)

        if self.args.tensorboard:
            self.writer.add_scalar("val-neg-loglikelihood", nll,
                                   global_step=steps)
            self.writer.add_scalar("val-precision", prec, global_step=steps)
            self.writer.add_scalar("val-recall", rec, global_step=steps)
            self.writer.add_scalar("val-f1", f1, global_step=steps)

        del preds_all, targets_all

    def save_checkpoint(self, epochs=None, iters=None, steps=None):
        logging.info("Saving checkpoint...")
        if isinstance(self.model, nn.DataParallel):
            module = self.model.module
        else:
            module = self.model
        state_dict = module.state_dict()

        if epochs is not None:
            name = f"ckpt-e{epochs:02d}"
        elif iters is not None:
            name = f"ckpt-i{iters:06d}"
        else:
            name = f"ckpt-s{steps:08d}"

        save_path = os.path.join(args.save_dir, name)
        torch.save(state_dict, save_path)
        logging.info(f"Checkpoint saved to '{save_path}'.")

    def on_iter_complete(self, loss, local_iter, global_iter, global_step):
        if self.args.tensorboard:
            self.writer.add_scalar("neg-loglikelihood", loss,
                                   global_step=global_step)

        for period_checker, func in self.repeatables.items():
            if period_checker(iters=global_iter, steps=global_step):
                func(iters=global_iter, steps=global_step)

    def on_epoch_complete(self, epoch_idx, global_iter, global_step):
        for period_checker, func in self.repeatables.items():
            if period_checker(epochs=epoch_idx,
                              iters=global_iter,
                              steps=global_step):
                func(epochs=epoch_idx,
                     iters=global_iter,
                     steps=global_step)


def check_arguments(args):
    num_inputs = len(args.input_path)

    assert num_inputs > 0, \
        "At least one input file must be specified."

    defaults = {
        "wordembed-type": "none",
        "wordembed-path": None,
        "wordembed-freeze": False,
        "word-dim": 300,
    }

    # default values for list type arguments
    for attr, default in defaults.items():
        attr_name = attr.replace("-", "_")
        if getattr(args, attr_name) is None:
            setattr(args, attr_name, [default] * num_inputs)

    # check if input counts are correct
    for attr in defaults:
        attr_name = attr.replace("-", "_")
        assert len(getattr(args, attr_name)) == num_inputs, \
            f"--{attr} must be specified as many as inputs. " \
            f"specified: {len(getattr(args, attr_name))}, required: {num_inputs}"

    assert 0.0 < args.val_ratio < 1.0, \
        "Specify a valid validation ratio."

    # ensure that the save-dir exists
    os.makedirs(args.save_dir, exist_ok=True)


def main(args):
    logging.basicConfig(level=logging.INFO)
    check_arguments(args)

    logging.info("Creating vocabulary...")
    input_vocabs = []

    for input in args.input_path:
        vocab = utils.Vocabulary()
        words = utils.FileReader(input).words()
        vocab.add("<pad>")
        utils.populate_vocab(words, vocab)
        input_vocabs.append(vocab)

    label_vocab = utils.Vocabulary()
    label_vocab.add("<pad>")
    words = utils.FileReader(args.label_path).words()
    utils.populate_vocab(words, label_vocab)

    for i, input_vocab in enumerate(input_vocabs):
        vocab_path = os.path.join(args.save_dir,
                                  f"vocab-input{i + 1:02d}.pkl")
        pickle.dump(input_vocab, open(vocab_path, "wb"))
    vocab_path = os.path.join(args.save_dir, f"vocab-label.pkl")
    pickle.dump(label_vocab, open(vocab_path, "wb"))

    logging.info("Initializing model...")
    crf = M.CRF(len(label_vocab))
    model = M.LSTMCRF(
        crf=crf,
        vocab_sizes=[len(v) for v in input_vocabs],
        word_dims=args.word_dim,
        hidden_dim=args.lstm_dim,
        layers=args.lstm_layers,
        dropout_prob=args.dropout_prob,
        bidirectional=args.bidirectional
    )
    model.reset_parameters()
    if args.gpu:
        gpu_main = args.gpu[0]
        model = model.cuda(gpu_main)
    params = sum(np.prod(p.size()) for p in model.parameters())
    logging.info(f"Number of parameters: {params}")

    logging.info("Loading word embeddings...")
    for vocab, we_type, we_path, we_freeze, emb in \
            zip(input_vocabs, args.wordembed_type, args.wordembed_path,
                args.wordembed_freeze, model.embeddings):
        if we_type == "glove":
            assert we_path is not None
            load_glove_embeddings(emb, vocab, we_path)
        elif we_type == "fasttext":
            assert we_path is not None
            assert args.fasttext_path is not None
            load_fasttext_embeddings(emb, vocab,
                                     fasttext_path=args.fasttext_path,
                                     embedding_path=we_path)
        elif we_type == "none":
            pass
        else:
            raise ValueError(f"Unrecognized word embedding "
                             f"type: {we_type}")

        if we_freeze:
            emb.weight.requires_grad = False

    # Copying configuration file to save directory if config file is specified.
    if args.config:
        config_path = os.path.join(args.save_dir, os.path.basename(args.config))
        shutil.copy(args.config, config_path)

    def create_dataloader(dataset):
        return D.MultiSentWordDataLoader(
            dataset=dataset,
            vocabs=input_vocabs + [label_vocab],
            batch_size=args.batch_size,
            shuffle=args.shuffle,
            tensor_lens=True,
            num_workers=len(args.gpu) if args.gpu is not None else 1,
            pin_memory=True
        )

    dataset = D.MultiSentWordDataset(*args.input_path, args.label_path)

    if args.val:
        vr = args.val_ratio
        train_dataset, val_dataset = dataset.split(1 - vr, vr,
                                                   shuffle=args.shuffle)
    else:
        train_dataset, val_dataset = dataset, None

    train_dataloader = create_dataloader(train_dataset)

    if val_dataset is not None:
        val_dataloader = create_dataloader(val_dataset)
    else:
        val_dataloader = None

    logging.info("Beginning training...")
    trainer = LSTMCRFTrainer(
        sargs=args,
        input_vocabs=input_vocabs,
        label_vocab=label_vocab,
        val_data=val_dataloader,
        model=model,
        epochs=args.epochs,
        gpus=args.gpu
    )
    trainer.train(train_dataloader,
                  data_size=len(train_dataset))

    logging.info("Done!")


if __name__ == '__main__':
    args = parser.parse_args()
    if args.tensorboard:
        import tensorboard as T
    main(args)