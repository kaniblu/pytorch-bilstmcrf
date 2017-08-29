import io
import os
import pickle

import tqdm
import torch
from torch.autograd import Variable
import torch.utils.data as tchdata
import numpy as np

from model import BiLSTMCRF
from utils import ensure_dir_exists
from utils.vocab import Vocabulary
from utils.argparser import ArgParser
from utils.argparser import path
from utils.preprocessor import Preprocessor
from utils.generator import SentenceGenerator
from utils.generator import TextFileReader


def parse_args():
    parser = ArgParser(allow_config=True)
    parser.add("--ckpt_path", type=path, required=True)
    parser.add("--feats_path", type=path, action="append", required=True)
    parser.add("--feats_vocab", type=path, action="append", required=True)
    parser.add("--labels_vocab", type=path, required=True)
    parser.add("--sent_tags", action="store_true", default=False)
    parser.add("--save_dir", type=path, required=True)
    parser.add("--batch_size", type=int, default=32)
    parser.add("--max_length", type=int, default=1e10)
    parser.add("--gpu", action="store_true", default=False)

    group = parser.add_group("Model Parameters")
    group.add("--word_dim", type=int, action="append", required=True)
    group.add("--hidden_dim", type=int, required=True)

    args = parser.parse_args()

    return args


def load_data(data_path):
    with io.open(data_path, "r", encoding="utf-8") as f:
        sents = f.readlines()

    sents = [sent.strip() for sent in sents]
    sents = [sent for sent in sents if sent]

    return sents


def create_data_loader(sents, preprocessor, batch_size):
    def _collate_fn(batch):
        lens = torch.LongTensor([len(s) + 1 for s in batch])
        batch = torch.LongTensor(preprocessor(batch))

        lens, idx = torch.sort(lens, dim=0, descending=True)
        batch = batch[idx]

        batch = batch.pin_memory()

        return batch, lens

    data_loader = tchdata.DataLoader(sents, batch_size, collate_fn=_collate_fn)
    return data_loader


def prepare_batch(xs, lens, gpu=True):
    lens, idx = torch.sort(lens, 0, True)
    _, ridx = torch.sort(idx, 0)
    idx_exp = idx.unsqueeze(0).unsqueeze(-1).expand_as(xs)
    xs = torch.gather(xs, 1, idx_exp)

    xs = Variable(xs, volatile=True)
    lens = Variable(lens, volatile=True)
    ridx = Variable(ridx, volatile=True)

    if gpu:
        xs = xs.cuda()
        lens = lens.cuda()
        ridx = ridx.cuda()

    return xs, lens, ridx


def to_sent(vocab, seq):
    return " ".join(vocab.i2f[x] if x in vocab.i2f else vocab.unk
                    for x in seq)


def predict(model, _data_loader_fn):
    all_scores, all_paths = [], []
    loaders = _data_loader_fn()
    t = tqdm.tqdm()

    for data in zip(*loaders):
        xs = torch.cat([x[0].unsqueeze(0) for x in data])
        lens = data[0][1]

        xs, lens, ridx = prepare_batch(xs, lens, model.is_cuda)

        logits = model._forward_bilstm(xs, lens)
        scores, paths = model.crf.viterbi_decode(logits, lens)

        scores_n = torch.index_select(scores, 0, ridx)
        paths_n = torch.index_select(paths, 0, ridx)
        lens = torch.index_select(lens, 0, ridx)

        lens = lens.cpu().data.tolist()
        scores_n = scores_n.cpu().data.tolist()
        paths_n = paths_n.cpu().data.tolist()

        paths_n = [path[:l] for path, l in zip(paths_n, lens)]

        all_scores.extend(scores_n)
        all_paths.extend(paths_n)

        t.update(len(lens))

    return all_scores, all_paths


def _load_vocab(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(scores, paths, save_dir):
    score_path = os.path.join(save_dir, "scores.txt")
    pred_path = os.path.join(save_dir, "preds.txt")

    with open(score_path, "w") as f:
        f.write("\n".join(map(str, scores)))

    with open(pred_path, "w") as f:
        f.write("\n".join(paths))


def main():
    args = parse_args()

    ensure_dir_exists(args.save_dir)

    print("Loading vocabulary...")
    feats_vocabs = [_load_vocab(path) for path in args.feats_vocab]
    labels_vocab = _load_vocab(args.labels_vocab)

    print("Loading model...")
    model_cls = BiLSTMCRF
    model = model_cls(feats_vocabs, labels_vocab,
                      word_dims=args.word_dim,
                      hidden_dim=args.hidden_dim,
                      dropout_prob=0)

    model.load_state_dict(
        torch.load(args.ckpt_path), map_location=lambda storage, loc: storage)

    if args.gpu:
        model = model.cuda()

    print("Predicting...")

    def _data_loader_fn():
        feats_preps = [Preprocessor(vocab) for vocab in feats_vocabs]
        feats_readers = [TextFileReader(path) for path in args.feats_path]

        feats_gen = [SentenceGenerator(reader, vocab, args.batch_size,
                                       max_length=args.max_length,
                                       preprocessor=prep,
                                       allow_residual=True)
                     for reader, vocab, prep in
                     zip(feats_readers, feats_vocabs, feats_preps)]

        return feats_gen

    scores, paths = predict(model, _data_loader_fn)

    if not args.sent_tags:
        paths = [path[1:-1] for path in paths]

    paths = [to_sent(labels_vocab, path) for path in paths]

    print("Saving results...")
    _save(scores, paths, args.save_dir)

    print("Done!")


if __name__ == '__main__':
    main()
