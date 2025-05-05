# AI Upskilling Workshop: Bioacoustics track

Welcome! In this workshop, we will be learning the basics of transfer learning by training frog species classifiers using audio embeddings from open-source bird species classifiers. Cool!

## 0. Environment setup

We will provide you with a login to a virtual machine managed by CyVerse. Once logged in, run the following commands to get an environment up and running:

**1. Install miniconda**, which we will use to manage Python environments:
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
chmod +x Miniconda3-latest-Linux-x86_64.sh
./Miniconda3-latest-Linux-x86_64.sh
```
Say "yes" to everything, and use all the defaults. If you accidentally say no to something, you can run it again with `./Miniconda3-latest-Linux-x86_64.sh -u` to fix it.

**2. Install perch-hoplite and dependencies**
```
source ~/.bashrc
conda create -n hoplite python=3.10
conda activate hoplite
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
pip install absl-py requests 'tensorflow[and-cuda]<2.16' tensorflow-hub ipywidgets etils
pip install git+https://github.com/google-research/perch-hoplite.git
git clone https://github.com/google-research/perch-hoplite.git
```

Once this is good to go, we recommend installing VS Code locally and connecting via SSH. See slides 19, 20, 33, and 34 [here](https://docs.google.com/presentation/d/1VFkmj5dvtlnziBOFM9GO4PEcfhcPrv7tfBaSo6uEOWg/edit#slide=id.g2680127c5bb_0_88) for a walkthrough.

## 1. Intro to perch-hoplite

Copy data from shared drive to your local drive:

`TODO`

Run through the steps in `perch_hoplite/agile/1_embed_audio_v2.ipynb`.

## 2. Data exploration

Complete the `01_anuraset_explore.ipynb` notebook.

## 3. Intro to Numpy

TODO

## 4. Training linear models

Complete the `03_anuraset_train.ipynb` notebook.

## The rest is yet to come!
