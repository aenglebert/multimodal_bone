# VLP training scripts for Self-supervised Vision-Language Deep Learning for Bone Radiography Analysis

(Paper available soon)

## Installation

Install requirements with `pip install -r requirements.txt`

## Configuration

This repository uses hydra configuration files in the "config" folder.

The main change that needs to be made is to adapt to your dataset.
See `config/dataset/lightning_wds_paired_single` and `data/wds_doc_rx.py` to see how we made it for our dataset.

We used the [webdataset](https://webdataset.github.io/webdataset/) library to load our data using a structure as follow inside of the shards:
```
patient1_study1.txt
patient1_study1.0.jpg
patient1_study2.txt
patient1_study2.0.jpg
patient1_study2.1.jpg
patient2_study3.txt
patient2_study3.0.jpg
patient2_study3.1.jpg
patient2_study3.2.jpg
```
Confer [webdataset documentation](https://webdataset.github.io/webdataset/) on how to create shards for your own dataset.

The process of curating our dataset is described in the Dataset_creation.pdf file.

## Training

To launch the VLP process, use the pretrain.py script
`python pretrain.py`

Using hydra, you can also dynamically override parameters, exemple:

`python pretrain.py batch_size=64 text_model=mluke`

