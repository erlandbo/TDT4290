# ResNet

This model uses the pretrained [pytorch ResNet34 model](https://pytorch.org/hub/pytorch_vision_resnet/)
for regression analysis on mel spectograms of soud data.
This model does not take time into account and only looks
at audio samples with vehicles in them.

The other versions of the ResNet (18, 50, 101, 152) were tried,
but ResNet 34 was the most accurate on the few epochs that we have run.


## Data

**Data pipeline:**
1. Split into 2s audio samples
2. Convert audio sample into MEL Spectogram
3. Convert MEL Spectogram into image to be used py model



## Results
| Dimension                  | MAE / Accuracy         |   Epochs runned |
|----------------------------|-----------------------:|----------------:|
| width (regression)         | 16 - 18  (MAE)         |  50             |
| height (regression)        | 22 - 24  (MAE)         |  50             |
| duration (regression)      | 0.16 - 0.17  (MAE)     |  50             |
| 3 classes (classification) |     0.876   (Accuracy) |  50             |
| 5 classes (classification) |  0.34-0.41  (Accuracy) |  50             |

## Files / Scripts 

### Preprocess data 
`preprocess_data.ipynb`
Notebook for preprocessing data and audio for this ResNet model.

This notebook uses lidar data and audio 
recordings found in `../../data/` folder.

### Audio classification
`Audio_classification.ipynb`
Classification using Pytorch and Librosa 


### Audio regression
`Audio_regression.ipynb`
Regression using Pytorch and Librosa

