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

    python -m utils.vocab --input_dir sents.txt --vocab_path vocab-sents.pkl --cutoff 30000

Use the vocabulary file for new training instances:

    python train.py --feats_path sents.txt --feats_vocab vocab-sents.pkl ...

More options are available through argparse help.

    python train.py --help

All options could be saved to a separate config file (either in json or yaml).

    python train.py --config train.yml

Models could handle multiple features

    python train.py --feats_path sents.txt --feats_path pos.txt --feats_vocab vocab-sents.pkl --feats_vocab vocab-pos.pkl ...

## Prediction ##

    python predict.py --model_path ...