# VAEBERT: Generating Shapes from Captions using Variational Auto Encoders and BERT

Code repository for the COMPSCI 674 course project titled "VAEBERT: Generating Shapes from Captions using Variational Auto Encoders and BERT" for Spring 2022. This project is heavily inspired from ["AI for 3D Generative Design"](https://blog.insightdatascience.com/ai-for-3d-generative-design-17503d0b3943) by Tyler Habowski ([repo](https://github.com/starstorms9/shape)). The technical report is available [here](https://gzhihongwei.github.io/files/munteanu2022vaebert.pdf).

## Overview

- [Installation](#installation)
- [Preprocessing](#preprocessing)
- [Training](#training)
- [Testing/Inference](#testinginference)
- [Citing](#citing)

## Installation

First, clone the repository.

```bash
git clone https://github.com/gzhihongwei/vaebert.git
```

Then, create a new python3 virtual environment in the root directory of the repo.

```bash
cd vaebert
python3 -m venv venv
```

Then install all dependencies (including this repo, which is installable under the `vaebert` package).

```bash
source venv/bin/activate
pip3 install -r requirements.txt
pip3 install -e .  # Installs vaebert as a package so that relative imports work
deactivate
```

## Preprocessing

1. Get a ShapeNet account [here](https://shapenet.org/).
1. Once given access to ShapeNet, download `ShapeNetCore.v2.zip` and place it under the `shapenet/`, which you should create under the root directory of this repo.
1. Download `mid2desc.pkl` from the 3D Generative Design shape repository (https://github.com/starstorms9/shape) [here](https://github.com/starstorms9/shape/blob/master/data/mid2desc.pkl?raw=true) and also place the file under the `shapenet/` directory.
1. Finally, run `python3 preprocess.py` to obtain all of the necessary preprocessed data files for training.

After running this procedure, the repo should be in the below structure:

```
vaebert
├─ .gitignore
├─ README.md
├─ binvox_rw.py
├─ preprocess.py
├─ requirements.txt
├─ setup.py
├─ shapenet
│  ├─ ShapeNetCore.v2.zip
│  ├─ mid2desc.pkl
│  ├─ partnet_data.h5
│  ├─ train_indexes.json
│  └─ test_indexes.json
├─ test.py
├─ vaebert
│  ├─ __init__.py
│  ├─ data.py
│  ├─ metrics.py
│  ├─ text
│  │  ├─ __init__.py
│  │  ├─ bert
│  │  │  ├─ __init__.py
│  │  │  ├─ bert_encoder.py
│  │  │  └─ train.py
│  │  ├─ data.py
│  │  └─ gru
│  │     ├─ __init__.py
│  │     ├─ gru_encoder.py
│  │     └─ train.py
│  ├─ vae
│  │  ├─ __init__.py
│  │  ├─ data.py
│  │  ├─ encode.py
│  │  ├─ train.py
│  │  └─ vae.py
│  └─ visualize.py
└─ visualize_caption.py
```

Now, you are ready to train!

## Training

1. Run `python3 vaebert/vae/train.py` to train the VAE. By default this will run for 20 epochs and save checkpoints to the `vaebert/vae/checkpoints/` folder.
1. Run `python3 vaebert/vae/encode.py -checkpoint vaebert/vae/checkpoints/epoch20.pt` to encode the voxelizations into latent vectors given the trained VAE.
1. Run `python3 vaebert/text/bert/train.py` to train the BERT text encoder model. By default this will train for 20 epochs and save checkpoints to the `vaebert/text/bert/checkpoints/` folder.
1. Run `python3 vaebert/text/gru/train.py` to train the Bi-GRU text encoder model. By default this will train for 1000 epochs and save checkpoints to the `vaebert/text/gru/checkpoints/` folder.

## Testing/Inference

1. Run `python3 test.py -checkpoints path/to/bert_checkpoint.pt path/to/gru_checkpoint.pt path/to/vae_checkpoint.pt` to get the average chamfer distance for the VAE + BERT and VAE + Bi-GRU pipelines and visualize a test set example from each shape category for both pipelines.
2. Run `python3 visualize_caption.py -checkpoints path/to/bert_checkpoint.pt path/to/gru_checkpoint.pt path/to/vae_checkpoint.pt` to interactively submit custom captions to the VAE + BERT and VAE + Bi-GRU pipelines and get the output voxelization.

## Citing

If you found this code useful, please cite

```bibtex
@misc{munteanu2022vaebert,
    url = {https://gzhihongwei.github.io/files/munteanu2022vaebert.pdf},
    author = {Munteanu, Alexandru and Wei, Yi and Wei, George Z.},
    title = {VAEBERT: Generating Shapes from Captions using Variational Auto Encoders and BERT},
    publisher = {COMPSCI 674: Intelligent Visual Computing},
    year = {2022}
}
```
