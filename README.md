# AI Upskilling Workshop: Bioacoustics track

Welcome! In this workshop, we will be learning the basics of transfer learning by training frog species classifiers using audio embeddings from open-source bird species classifiers. Cool!

## 0. Environment setup

1. We recommend installing VS Code locally, and then connecting to the remote machines that will be provided by CyVerse for the workshop. 
You can follow the guide here to get set up, ignoring the bit in the middle about installing things on your own machine: 
[Environment setup](https://github.com/cv4ecology/cv4ecology.github.io/blob/main/ai-upskilling-workshop-2025-tools.md#recommended-software-installation-for-2025-ai-upskilling-workshop).

2. Create a conda environment on the remote machine and install the packages you'll need.
```
conda create -n hoplite python=3.10
conda activate hoplite
sudo apt-get update
sudo apt-get install libsndfile1 ffmpeg
pip install absl-py requests 'tensorflow-cpu<2.16' tensorflow-hub
```

3. Clone the `perch-hoplite` codebase.
```
git clone https://github.com/google-research/perch-hoplite.git
```

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
