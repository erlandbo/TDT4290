# SSED - Sliding-window Sound Event Detection

This repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES, with contributions by NINA - Norsk Institutt for NAturfroskning.

The codebase contains multiple convolutional backbones, and is (almost) fully configurable in both training and inference through yaml specified configurations.
This codebase should be easily extendable and reusable for most sound event detection tasks, and I hope you get good use of it, just remember I've licensed it under the MIT License, and a lot of the other stuff that I've used used is licensed under several other licenses as well (can be found in the LICENSES subdirectory), read up on those so you don't get hit with a C&D.

# Install dependencies

(Recommended) use pip.
```bash
mkdir env
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```


Another option is to create a conda environment that will contain python 3 with dependencies:
```
conda env create -f ssed_env.yml
conda activate ssed_tdt4290
```


## Troubleshooting
- Pytorch: You may want to install a specific torch version https://pytorch.org/get-started/locally/


# Datasets


The model has been trained and tested on the following recordings and lidar-data
- audio_22092022.WAV and lidar_data_with_audio_timestamps_22.csv
- audio_27092022_1.WAV and lidar_data_with_audio_timestamps_27_1.csv
- audio_27092022_2.WAV and lidar_data_with_audio_timestamps_27_2.csv
- audio_01112022_1.WAV and lidar_data_with_audio_timestamps_nov_01_1.csv
- audio_01112022_2.WAV and lidar_data_with_audio_timestamps_nov_01_2.csv

The recordings from 21.oktober and 28.october were concidered corrupt because of high picthed noise. 

You can build the dataset with all the recordings or hold-out one for inference.
   
## Recording 22.september 2022
Corresponding files:
- audio_22092022.WAV and lidar_data_with_audio_timestamps_22.csv

The recording were taken from hours 10 to 12.37. There were no precipitation and approximately 11-13 celsius degrees. 


| Class   | Train   | Val  | Test|
| ------- | --- | --- |---|
| Small | 29 | 3 |3|
| Medium | 368 | 29 |55|
| Large | 63 | 8 |9|


## Recording 27.september 2022
Corresponding files:
- audio_27092022_1.WAV and lidar_data_with_audio_timestamps_27_1.csv
- audio_27092022_2.WAV and lidar_data_with_audio_timestamps_27_2.csv


The recording were taken from hours 12 to 17. There were 10.9mm precipitation that day but not during the recording, and approximately 10-11 celsius degrees. 


| Class   | Train   | Val  | Test|
| ------- | --- | --- |---|
| Small | 72 | 18 |16|
| Medium | 838 | 115 |150|
| Large | 98 | 13 |6|


## Recording 01.November 2022
Corresponding files:
- audio_01112022_1.WAV and lidar_data_with_audio_timestamps_nov_01_1.csv
- audio_01112022_2.WAV and lidar_data_with_audio_timestamps_nov_01_2.csv


The recording were taken from hours 9 to 15. There were no precipitation that day, and approximately 5-9 celsius degrees. 


| Class   | Train   | Val  | Test|
| ------- | --- | --- |---|
| Small | 28 | 3 |4|
| Medium | 784 | 119 |138|
| Large | 119 | 16 |13|

Notice that some classes are underepresented so the chosen dataset-split may affect the results.

# Build dataset

```bash
python build_dataset.py -a <audio> -d <data> -c <class> 
```

For example
```bash
python build_dataset.py -a ../data/audio_22092022.wav -d ../data/lidar_data_22092022.csv -c class_1 
```
The class-argument corresponds to the class-column in the .csv


The command will split the 80% of the dataset to training, 10% of the dataset to validation and 10% of the dataset to testing. 


# Train and inference
You may require significant hardware-resources for training the model. A Nvidia Tesla T4 GPU or similar is sufficient.

To train the model:
```bash
python train.py configs/default.yaml
```

This will train the model for a chosen number of epochs and validate for a chosen number of epochs (checkout the configs in the /classifier/config/defaults.py). Finally after training, the best model from the validation will be loaded and the results from the testing will be printed. 

## Troubleshooting
- You may want to reduce the batch-size in classifier/config/defaults.py if you receive a *CUDA OUT OF MEMORY* error.


## Experimental
Then to run inferrence with the model you've trained:

```bash
python infer.py configs/default.yaml path/to/audio_file.wav
```

This trains an EfficientNet-b7 based model on the dataset I've added, with the basic config file, and runs inference based on the model you've trained. 

Notice you may want to experiment with windows-size and IoU threshold for better results.


