import os
import json

from utils.argparser import ArgParser
from utils.argparser import path
from utils import ensure_dir_exists


def normalize_pretag(pretag):
    if pretag == "S":
        return "B"
    elif pretag == "E":
        return "I"

    return pretag


def preprocess_labels(label_sents):
    data = []

    for sent in label_sents:
        for label in sent:
            tokens = label.split("-")

            if len(tokens) == 2:
                pretag = tokens[0]
                tag = tokens[1]

                pretag = normalize_pretag(pretag)
            elif len(tokens) == 1:
                pretag, tag = "O", "O"
            else:
                continue

            data.append((pretag, tag))

    return data


def load_file(path):
    with open(path, 'r') as fp:
        return [l.strip().split() for l in fp.readlines()]


def compute_f1(preds, y):
    prec = compute_precision(preds, y)
    rec = compute_precision(y, preds)

    f1 = 0
    if (rec + prec) > 0:
        f1 = 2.0 * prec * rec / (prec + rec)

    return (prec, rec, f1)


def compute_precision(guessed, correct):
    correctCount = 0
    count = 0

    idx = 0
    while idx < len(guessed):
        if guessed[idx][0] == 'B':  # A new chunk starts
            count += 1
            if guessed[idx] == correct[idx]:
                idx += 1
                correctlyFound = True

                while idx < len(guessed) and guessed[idx][0] == 'I':
                    if guessed[idx] != correct[idx]:
                        correctlyFound = False

                    idx += 1

                if idx < len(guessed):
                    if correct[idx][0] == 'I':
                        correctlyFound = False

                if correctlyFound:
                    correctCount += 1
            else:
                idx += 1
        else:
            idx += 1

    precision = 0
    if count > 0:
        precision = float(correctCount) / count

    return precision


def parse_args():
    parser = ArgParser(allow_config=True)
    parser.add("--pred_path", type=path, required=True)
    parser.add("--gold_path", type=path, required=True)
    parser.add("--out_path", type=path, required=True)

    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    ensure_dir_exists(args.out_path)

    print("Loading files...")
    preds = load_file(args.pred_path)
    golds = load_file(args.gold_path)
    preds = preprocess_labels(preds)
    golds = preprocess_labels(golds)

    print("Computing scores...")
    prec, rec, f1 = compute_f1(preds, golds)

    data = {
        "prec": prec,
        "rec": rec,
        "f1": f1
    }

    print("Saving...")
    with open(args.out_path, "w") as f:
        f.write(json.dumps(data))

    print("Done!")


if __name__ == '__main__':
    main()
