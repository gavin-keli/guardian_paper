Guardian model
===============
This repository contains our implementation of the paper published in IEEE Transactions on Dependable and Secure Computing, "Defend Data Poisoning Attacks on Voice Authentication".

[Paper link here](https://ieeexplore.ieee.org/abstract/document/10163863)


## Installation
First, clone the repository locally, create and activate a conda environment, and install the requirements :
```
$ git clone https://github.com/gavin-keli/guardian_paper.git
$ conda create -n guardian_paper python=3.7
$ conda activate guardian_paper
$ python setup.py
$ pip install -r requirements.txt
```


## Experiments

### Dataset
Our experiments are performed on the [LibriSpeech](http://www.openslr.org/12/).

The Speaker authentication model is based on [Deep Speaker: an End-to-End Neural Speaker Embedding System](https://arxiv.org/pdf/1705.02304.pdf)

Reference code: https://github.com/philipperemy/deep-speaker and https://github.com/Walleclipse/Deep_Speaker-speaker_recognition_system (Thanks to Philippe Rémy and Abuduweili) 

### Pre-Process
There are two steps to pre-process the dataset:

1. Extract fbanck from WAVs and save as NPY files
```
$ cd ./src
$ python pre_process_npy.py
```

2. Convert NPY files to Embeddings (2K/512 or 4K/1024)
```
$ python pre_process_embedding.py
```

Notice:
There is a small sample dataset in the repo, which includes 4 speakers. 669 and 711 are benign(normal) users; 597 and 770 are victims; 2812 and 2401 are two attackers.

### Training & Testing Deep Speaker
Please check Philippe Rémy and Abuduweili's code.

### Training 
First, you need to generate/save a guardian model
```
$ python save_model.py
$
$ 1122809848
```

To train the model run:
```
$ python train.py
$
$ Please enter the model ID: 1122809848
```

### Testing

To evaluate your own model:
```
$ python test.py
$
$ Please enter the name_training: 1122809848-100
```

Here, you just test the CNN model, not the whole Guardian system.


### Preparing for KNN

Save the CNN results into CSV files, and prepare for training KNN
```
$ python knn_stat.py
$
$ Please enter the name_training: 1122809848-100
```

### Training & Testing Guardian system

To fill in the last piece of the puzzle, you can use the Jupyter Notebook (***guardian.ipynb***) to train/test the whole Guardian system



## Results
Please check our [paper](https://ieeexplore.ieee.org/abstract/document/10163863).


## Contact
For any query regarding this repository, please contact:
- Ke(Gavin) Li: ke.li.1 [at] vanderbilt [dot] edu


## Citation
If you use this code in your research please use the following citation:
```bibtex
@article{guardian_paper, 
year = {2023}, 
title = {{Defend Data Poisoning Attacks on Voice Authentication}}, 
author = {Li, Ke and Baird, Cameron and Lin, Dan}, 
journal = {IEEE Transactions on Dependable and Secure Computing}, 
issn = {1545-5971}, 
doi = {10.1109/tdsc.2023.3289446}
}
```
