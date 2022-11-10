# TDT4290 Customer Driven Project  (2022 Fall)

| **Task**     | *Acoustic based vehicle detection and classification* |
|----|-----|
| **Group**    | 6       |
| **Customer** | [Q-FREE](https://www.q-free.com)  |

## Introduction

Electronic toll collection (ETC) has played a key role in
revolutionizing the flow of modern traffic.
In most ETC systems, vehicles are equipped with a radio transponder,
which can transmit the information about the vehicle to a receiver for
toll collection. Without a transponder (e.g. in foreign vehicles),
automatic toll collection required a more complex solution that
involves advanced sensor technology for vehicle identification and
dimension measurement.

This project aims to study the feasibility of replacing current expensive lidar-based
system for vehicle dimension measurement with an acoustic-based solution using microphones.
The project would involve setting up the necessary equipment to collect
and analyze the sound of passing vehicles.
The collected sound data will then be combined with measurement data from our existing
lidar-based system to develop and train machine learning models that are capable of deriving
vehicle's properties such as dimensions from just the acoustic (sound) signals.

# Repository
## Folder structure
```
TDT4290/
├─ data/
│  ├─ ...       # Lidar logs and audio recordings
├─ models/
│  ├─ mel_coef/ # Classical ML - Mel Coeffisient model
│  ├─ resnet/   # ResNet model
│  ├─ SSED/     # SSED Classification model
│  ├─ SSED-dim/ # SSED Regression model
│  ├─ FDY-CRNN.md
│  ├─ SSED.md
│  ├─ ...
├─ preprocess_data/
│  ├─ ...       # Notebook on audio preprocessing
├─ preprocess_lidar/
│  ├─ ...       # Notebook on lidar log processing
```


## Data
The `data/` folder contains the following types of files used by the models:
- Lidar logs
    1. Raw logs
    2. Processed logs matched with audio
- Audio files

The audio-data can be downloaded by script. The script must be given executable privileges, by e.g. chmod 
```
cd data
chmod +x download.sh
./download.sh
```

   
## Recording 22.september 2022
Corresponding files:
- audio_22092022.WAV and lidar_data_with_audio_timestamps_22.csv

| Class   | Train   | Val  | Test|
| ------- | --- | --- |---|
| Small | 29 | 3 |3|
| Medium | 344 | 29 |49|
| Large | 47 | 6 |9|


## Recording 27.september 2022
Corresponding files:
- audio_27092022_1.WAV and lidar_data_with_audio_timestamps_27_1.csv
- audio_27092022_2.WAV and lidar_data_with_audio_timestamps_27_2.csv


| Class   | Train   | Val  | Test|
| ------- | --- | --- |---|
| Small | 68 | 18 |16|
| Medium | 756 | 107 |142|
| Large | 78 | 11 |6|


## Recording 01.November 2022
Corresponding files:
- audio_01112022_1.WAV and lidar_data_with_audio_timestamps_nov_01_1.csv
- audio_01112022_2.WAV and lidar_data_with_audio_timestamps_nov_01_2.csv



| Class   | Train   | Val  | Test|
| ------- | --- | --- |---|
| Small | 28 | 3 |4|
| Medium | 710 | 105 |128|
| Large | 77 | 14 |9|



## Preprocessing
The `preprocess/` folder contains the necessary files
for preprocessing the lidar log and information
on audio preprocessing.

For more information on features used by the different models,
check out [Feature Exploration](./preprocess_data/feature_exploration.md).

## Models
Information on all the models examined can be found under the `models/` folder.

For more information about the different models,
check out [Model Overview](./models/README.md).

### SSED
This model uses sliding window for sound event detection (SED).
The model can detect when and what type of vehicle that is passing on a long audio recording. 

There is both a regression and classification variant of this model.
The classification one yielded the best result based on the limited
training we were able to do.

[Read more on SSED](./models/SSED.md)

[Read more on SSED regression on width and height](./models/SSED-dim.md)

### ResNet
This model uses a redisual network for doing classification and regression
on audio files of length 2 seconds.

[Read more on ResNet](./models/resnet/README.MD)

### Classical ML - Mel Coeffisients
This model uses a classical ML approach for regression on audio files by
extracting features like mel coeffisients on audio files with length of 2 seconds.

[Read more on MEL Coef](./models/mel_coef/README.MD)

### FDY-CRNN
This model looked promising for sound event detection (SED),
but we were unfortunately unable to apply it to our dataset.

[Read more on FDY-CRNN](./models/FDY-CRNN.md)
