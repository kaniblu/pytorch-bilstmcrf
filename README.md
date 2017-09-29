# BiLSTM-CRF on PyTorch #

An efficient BiLSTM-CRF implementation that can leverage batch operations on
GPUs.

This repository must be cloned *recursively* due to submodules.

    git clone https://github.com/kaniblu/pytorch-bilstmcrf --recursive

Install all required packages (other than pytorch) from utils/requirements.txt

    pip install -r requirements.txt

Run visdom server beforehand.

    python -m visdom.server

## Training ##

Create vocabulary per feature (e.g.):

    python -m utils.vocab --data_dir sents.txt --vocab_path vocab-sents.pkl --cutoff 30000

Use the vocabulary file for new training instances:

    python train.py --feats_path sents.txt --feats_vocab vocab-sents.pkl ...

More options are available through argparse help.

    python train.py --help

All options could be saved to a separate config file (either in json or yaml).

    python train.py --config train.yml

Models could handle multiple features

    python train.py --feats_path sents.txt --feats_path pos.txt --feats_vocab vocab-sents.pkl --feats_vocab vocab-pos.pkl ...

## Prediction ##

Predict tags with given features. Specify model path with `--ckpt_path` option. Model parameters should be identical to those that have been used to train it.

    python predict.py --ckpt_path ... --feats_path sents_test.txt --feats_path pos_test.txt --feats_vocab vocab-sents.pkl ... --save_dir ./output
    
Tagged file `preds.txt` and score file `scores.txt` will be written to `--save_dir` directory.

## Evaluation ##

Evaluate predictions with an answer set.

    python evaluate.py --pred_path ./output/preds.txt --gold_path .../tags_test.txt --out_path ./output/res.json
    
A simple json file containing the accuracy, precision and f1-score of the test will be written to `--out_path`.
