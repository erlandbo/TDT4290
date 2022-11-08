# SSED - Sliding-window Sound Event Detection

SSED - Sliding-window Sound Event Detection modified to output width, height.

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


The model has been trained and testes on the following recordings and lidar-data
- audio_22092022.WAV and lidar_data_with_audio_timestamps_22.csv
- audio_27092022_1.WAV and lidar_data_with_audio_timestamps_27_1.csv
- audio_27092022_2.WAV and lidar_data_with_audio_timestamps_27_2.csv
- audio_01112022_1.WAV and lidar_data_with_audio_timestamps_nov_01_1.csv
- audio_01112022_2.WAV and lidar_data_with_audio_timestamps_nov_01_2.csv

# Build dataset

```bash
python build_dataset.py -a <audio> -d <data>
```

For example
```bash
python build_dataset.py -a ../data/audio_22092022.wav -d ../data/lidar_data_22092022.csv
```

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



