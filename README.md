# TDT4290 Customer Driven Project  (2022 Fall)

## Introduction to acoustic-based vehicle dimension measurement

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


## Preproccesing


## Models
Information on all the models examined can be found under the `models/` folder.

### SSED
This model uses sliding window for sound event detection (SED).
The model can detect when and what type of vehicle that is passing on a long audio recording. 

There is both a regression and classification variant of this model.
The classification one yielded the best result based on the limited
training we were able to do.

[Read More](./models/SSED.md)

### ResNet
This model uses a redisual network for doing classification and regression
on audio files of length 2 seconds.

[Read More](./models/resnet/README.MD)

### Classical ML - Mel Coeffisients
This model uses a classical ML approach for regression on audio files by
extracting features like mel coeffisients on audio files with length of 2 seconds.

[Read More](./models/mel_coef/README.MD)

### FDY-CRNN
This model looked promising for sound event detection (SED),
but we were unfortunately unable to apply it to our dataset.

[Read More](./models/FDY-CRNN.md)