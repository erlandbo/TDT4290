# Classic machine learning on different audio features

This folder contains notebooks for data pre-processing, feature extraction from audio files 
and the result of applying different ML algorithms on the calculated features. 
This approach does not take time into consideration and rely on already sliced audio files. 

## Data

**Data pipeline:**
1. Split into 2s audio samples
2. Store split audio files into folder
3. Calculate zero crossing rate and mel coefficients 
4. calculate the duration it takes for a vehicle to pass the sensor


## Results
| Dimension                  | MAE / Accuracy        |
|----------------------------|-----------------------|
| width (regression)         | 16 - 18  (MAE)        |   
| width w time (regression)  | 7-8    (MAE)          |


## Findings 

Based on the results we are able to get a prediction which is approx. 6 cm better than just guessing the average width. 
If we include the time it takes for a car to pass the sensor the model is able to predict the width with MAE of 7-8 cm.  
Several different approaches were tried to estimate the duration but none were successful. 

## Files
Recommended to use pip.
### Pre-processing of data 
Pre-processing of the audio and lidar data is done in [create_data.ipynb](create_data.ipynb).
This notebook uses lidar data and audio 
recordings found in `../../data/` folder and stores split audio files 
in the folder "../../data/splitted_audio". 


### Width regression
In [get_features.ipynb](get_features.ipynb), regression is performed using Librosa and Pycaret on "width" with and without the duration of the passing car. 

**NB! Has to run on Python 3.8.2 to be able to run Pycaret.**
