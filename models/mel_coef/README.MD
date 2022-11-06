# CLassic machine learning on different audio features

This folder contains notebooks for data pre-processing, feature extraction from audio files 
and the result of applying different ML algorithms on the calculated features. 
This approach does not take time into consideration and rely on already sliced audio files. 

## Data

**Data pipeline:**
1. Split into 2s audio samples
2. store splitted audio files into folder
3. Calculate zero crossing rate and mel coefficients 
4. calculate the duration it takes for a vehicle to pass the sensor


## Results
| Dimension                  | MAE / Accuracy        |
|----------------------------|-----------------------|
| width (regression)         | 16 - 18  (MAE)        |   
| width w time (regression)  | 7-8    (MAE)          |


## Findings 

Based on the results we are not able to provide  a satesfactionary regressions result on width 
based on mel coefecients and zero crossing rate.
The prediction given by the models were only sligthly better than always preidcting the average. 
However, it appears that the time a vehicle use to pass the sensor is of high importance when attemping
to predict its width. Several different approaches were tried but none were succsessfull. 
