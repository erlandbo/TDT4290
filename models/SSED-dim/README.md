# SSED - Sliding-window Sound Event Detection

Source: [GitHub-link](https://github.com/bendikbo/SSED).

Master Thesis: [Thesis](https://github.com/bendikbo/SSED/blob/main/thesis.pdf).

SSED trained on width and height.

# Install dependencies

(Recommended) use pip.
```bash
mkdir env
python3 -m venv env
. env/bin/activate
pip install -r requirements.txt
```


Another option is to create a conda environment that will contain python 3 with dependencies:


## Troubleshooting
- Pytorch: You may want to install a specific torch version https://pytorch.org/get-started/locally/


# Datasets


The model has been trained and tested on the following recordings and lidar-data
- audio_22092022.WAV and lidar_data_with_audio_timestamps_22.csv
- audio_27092022_1.WAV and lidar_data_with_audio_timestamps_27_1.csv
- audio_27092022_2.WAV and lidar_data_with_audio_timestamps_27_2.csv
- audio_01112022_1.WAV and lidar_data_with_audio_timestamps_nov_01_1.csv
- audio_01112022_2.WAV and lidar_data_with_audio_timestamps_nov_01_2.csv

# Build dataset

```bash
python build_dataset.py -a <audio> -d <data> 
```

The audio-argument -a corresponds to the audio-recording .wav and the data-argument -d corresponds to the lidar-data .csv
The script expects a relative path to the data, so you must keep this convention.

For example
```bash
python build_dataset.py -a ../../data/audio_22092022.WAV -d ../../data/lidar_data_with_audio_timestamps_22.csv
python build_dataset.py -a ../../data/audio_27092022_1.WAV -d ../../data/lidar_data_with_audio_timestamps_27_1.csv
python build_dataset.py -a ../../data/audio_27092022_2.WAV -d ../../data/lidar_data_with_audio_timestamps_27_2.csv
python build_dataset.py -a ../../data/audio_01112022_1.WAV -d ../../data/lidar_data_with_audio_timestamps_nov_01_1.csv 
python build_dataset.py -a ../../data/audio_01112022_2.WAV -d ../../data/lidar_data_with_audio_timestamps_nov_01_2.csv
```

The command will split the 80% of the dataset to training, 10% of the dataset to validation and 10% of the dataset to testing. 


# Train and inference
You may require significant hardware-resources for training the model. A Nvidia Tesla T4 GPU or similar is sufficient.

To train the model:
```bash
python train.py configs/default.yaml
```

This will train the model for a chosen number of epochs and validate for a chosen number of epochs (checkout the configs in the /classifier/config/defaults.py). The model will print the targets and predictions on each validation epoch. 

## Troubleshooting
- You may want to reduce the batch-size in classifier/config/defaults.py if you receive a *CUDA OUT OF MEMORY* error.

