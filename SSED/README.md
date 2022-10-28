# SSED - Sliding-window Sound Event Detection

This repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES, with contributions by NINA - Norsk Institutt for NAturfroskning.

The codebase contains multiple convolutional backbones, and is (almost) fully configurable in both training and inference through yaml specified configurations.
This codebase should be easily extendable and reusable for most sound event detection tasks, and I hope you get good use of it, just remember I've licensed it under the MIT License, and a lot of the other stuff that I've used used is licensed under several other licenses as well (can be found in the LICENSES subdirectory), read up on those so you don't get hit with a C&D.


# Example application with existing backbones and datasets

So the full list of bash terminal commands to train a (somewhat) state of the art sound event detection system for the bird sounds in the dataset should be as simple as:


(Recommended) Create a conda environment that will contain python 3 with dependencies:
```
conda env create -f ssed_env.yml
conda activate ssed_tdt4290
```


Another option is to use pip, assuming you have a somewhat new version of python3 already installed, and has installed the virtualenv package.

```bash
mkdir env
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```


## Troubleshooting
- Pytorch: You may want to install a specific torch version https://pytorch.org/get-started/locally/

# How to create "state of the art model" with your own data
Okay, so you have a problem where you actually need to create your own * *state of the art* * model for sound event detection, because of reasons. No worries! I'll take you through the steps of doing it right here. BTW, this section is still being written/developed, email me if you have any questions. The project is setup for classification on 3 vehicle classes, but if you want to change the number of classes:

Change the following sections

- classifier/config/defaults.py
- classifier/data/datasets/__init__.py
- classifier/data/datasets/kauto5cls.py
- configs/default.yaml

You may want to reduce the batch-size in classifier/config/defaults.py if you receive a *CUDA OUT OF MEMORY* error. 
# Build dataset

```bash
python build_dataset.py -a <audio> -d <data> -c <class> 
```

For example
```bash
python build_dataset.py -a audio_22092022.wav -d lidar_data_22092022.csv -c class_1 
```
The class-argument corresponds to the class-column in the .csv

# Train and inference
You may require significant hardware-resources for training the model. A Nvidia Tesla T4 GPU or similar is sufficient.

To train the model:
```bash
python train.py configs/default.yaml
```

Experimental
Then to run inferrence with the model you've trained:

```bash
python infer.py configs/default.yaml path/to/audio_file.wav
```

This trains an EfficientNet-b7 based model on the dataset I've added, with the basic config file, and runs inference based on the model you've trained.



