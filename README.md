# BiLSTM-CRF on PyTorch #

An efficient BiLSTM-CRF implementation that leverages mini-batch operations on multiple GPUs.

Tested on the latest PyTorch Version (0.3.0) and Python 3.5+.

The latest training code utilizes GPU better and provides options for data parallization across multiple GPUs using `torch.nn.DataParallel` functionality.

## Requirements ##

Install all required packages (other than pytorch) from `requirements.txt`

    pip install -r requirements.txt

Optionally, standalone tensorboard can be installed from `https://github.com/dmlc/tensorboard` for visualization and plotting capabilities.

## Training ##

Prepare data first. Data must be supplied with separate text files for each input or target label type. Each line contains a single sequence, and each pair of tokens are separated with a space. For example, for the task of Named Entity Recognition using words and Part-of-Speech tags, the input and label files might be prepared as follows:

    (sents.txt)
    the fat rat sat on a mat
    the cat sat on a mat
    ...
    
    (pos.txt)
    det adj noun verb prep det noun
    det noun verb prep det noun
    ...
    
    (labels.txt)
    O O B-Animal O O O B-Object
    O B-Animal O O O O B-Object
    ...
    
Then above input and label files are provided to `train.py` using `--input-path` and `--label-path` respectively.

    python train.py --input-path sents.txt --input-path pos.txt --label-path labels.txt ...

You might need to setup several more parameters in order to make it work. Checkout `examples/atis` for an example of training a simple BiLSTM-CRF model with ATIS dataset. Run `python preprocess.py` at the example directory to convert to the dataset to`train.py`-friendly format, then run 

    python ../../train.py --config train-atis.yml`
 
 to see a running example. The example configuration assumes that standalone tensorboard is installed (you could turn it off in the configuration file).
 
 For more information on the configurations, check out `python train.py --help`.

## Prediction ##

`TODO`

## Evaluation ##

`TODO`
