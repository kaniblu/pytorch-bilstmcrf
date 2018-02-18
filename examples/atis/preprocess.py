import os
import pickle

import yaap


parser = yaap.ArgParser()
parser.add_argument("--pickle-path", type=yaap.path, default="atis.pkl")
parser.add_argument("--save-dir", type=yaap.path, default="data")


def to_text(words, nes, labels, i2w, i2n, i2l, save_dir):
    word_path = os.path.join(save_dir, "words.txt")
    nes_path = os.path.join(save_dir, "nes.txt")
    label_path = os.path.join(save_dir, "labels.txt")
    word_f = open(word_path, "w")
    nes_f = open(nes_path, "w")
    label_f = open(label_path, "w")

    for w, n, l in zip(words, nes, labels):
        word_f.write(" ".join(i2w[x] for x in w) + "\n")
        nes_f.write(" ".join(i2n[x] for x in n) + "\n")
        label_f.write(" ".join(i2l[x] for x in l) + "\n")


def preprocess(args):
    os.makedirs(args.save_dir, exist_ok=True)
    data = pickle.load(open(args.pickle_path, "rb"),
                       encoding="latin1")
    train, test, vocabs = data
    w2i, n2i, l2i = vocabs["words2idx"], vocabs["tables2idx"], vocabs["labels2idx"]
    i2w, i2n, i2l = [{v: k for k, v in vocab.items()}
                     for vocab in [w2i, n2i, l2i]]

    for split in ["train", "test"]:
        split_data = eval(split)
        dir = os.path.join(args.save_dir, split)
        os.makedirs(dir, exist_ok=True)
        to_text(*split_data, i2w, i2n, i2l, dir)


if __name__ == "__main__":
    args = parser.parse_args()
    preprocess(args)