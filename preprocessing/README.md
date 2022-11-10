# Preprocessing
This section discusses the preprocessing and exploration of the audio datasets and the lidar data datasets. 

## Preprocessing of lidar data

The lidar data is loaded as a Pandas data frame and filtered to only contain the enter_date, enter_time, leave_time, y0, y1 and height. From these values we compute the width of the vehicle (from y0 and y1), front_area (using the calculated width and height), and corresponding datetime columns from the date and time columns. This is done with the `parse_lidar` function in [parse_lidar_data.py](src/parse_lidar_data.py)

Audio timestamps are added to the lidar data, by telling the `timestamp_lidar` (in [timestamp_lidar_data](src/timestamp_lidar_data.py)) function the start time of the audio clip, which synchronizes the lidar data and sound data. The lidar data timestamps the vehicles just in the timeframe they pass the lidar sensor, which is often just 0.2 seconds. To make it easier for the machine learning models to learn, each timestamp are extended to have a constant length decided by the user, by default 2 seconds.

Each vehicle is classified with the `classify_lidar` function in [classify_lidar_data](src/classify_lidar_data.py). The classes can be passed dynamically into the function, but is by default the classes the team found the most reasonable based on the analysis and graphs in [lidar_data_parsing](lidar_data_parsing.ipynb). The classes denote the largest width that can be considered that class.

Each function of the preprocessing of the lidar data returns a pandas data frame with the new columns. All the function can be run in sequence using `label_lidar` from [label_lidar.py](src/label_lidar_data.py). You can see how this functino, as well as the previously mentioned functions are used in [label_lidar.ipynb](label_lidar.ipynb). This notebooks was also used to verify that each timestamp contained the sound of a passing vehicle.

## Preprocessing of audio data

Exploriatory data analysis can be found in [eda-q-free.ipynb](eda-q-free.ipynb) where the audio of a motor cycle, car, normal bus and metrobus are inspected. Based on the analysis it is safe to assume that it should be possible to classify the vehicles into different categories based on their audio signature. 


## Feature exploration
 
By performing feature exploration we were able to detect several important features that could be used in classefication and regression task. 


### Audio features

| Feature            | Model                | Description                                                                                                                                                                                     |
|--------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MFCC               | forest/decision tree | mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. |
| MEL spectrogram    | RESNET/SSD           | A spectrogram where the frequencies are converted to the mel scale.                                                                                                                             |
| Zero crossing rate | forest/decision tree | Number of times the audio signal crosses 0.                                                                                                                                                     |
| RMS energy         | Vehicle detection    | Root mean square of the total magnitude of the signal. Correnspond to how loud the signal is.                                                                                                   |
| Spectral Centroid  | Not used             | Indicates at which frequency the energy of a spectrum is centered upon.                                                                                                                         |
### Feature usage

#### SSD and resnet34
Both models use the mel spectrogram of each passing vehicle to perform image recognition. The resnet takes processed audio files, while the SSD consumes a audio and perform vehicle detection as well as image recognition. 

#### Forest / decision tree

The decision tree consumes a dataframe consisting of the first 20 Mel-frequency cepstral coefficients and the zero crossing rate of the audio file. The features are calculated using the librosa library. 
 
### Feature exploration
 
By performing feature exploration we were able to detect several important features that could be used in classefication and regression task. 


### Audio features

| Feature            | Model                | Description                                                                                                                                                                                     |
|--------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MFCC               | forest/decision tree | mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. |
| MEL spectrogram    | RESNET/SSD           | A spectrogram where the frequencies are converted to the mel scale.                                                                                                                             |
| Zero crossing rate | forest/decision tree | Number of times the audio signal crosses 0.                                                                                                                                                     |
| RMS energy         | Vehicle detection    | Root mean square of the total magnitude of the signal. Correnspond to how loud the signal is.                                                                                                   |
| Spectral Centroid  | Not used             | Indicates at which frequency the energy of a spectrum is centered upon.                                                                                                                         |

#### SSD and resnet34
Both models use the mel spectrogram of each passing vehicle to perform image recognition. The resnet takes processed audio files, while the SSD consumes a audio and perform vehicle detection as well as image recognition. 

#### Forest / decision tree

The decision tree consumes a dataframe consisting of the first 20 Mel-frequency cepstral coefficients and the zero crossing rate of the audio file. The features are calculated using the librosa library. 
 