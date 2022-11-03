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
| 3 classes (classification) |             (Accuracy) | |
| 5 classes (classification) |             (Accuracy) | |

## Files / Scripts 

### Preprocess data 
Preprocess data and audio in preprocess_data.ipynb.
This notebook uses lidar data and audio 
recordings found in `../../data/` folder.

### Audio classification
Classification using Pytorch and Librosa
Run instructions for classification

```bash
python audio_classification.py -c <class>
```

Classification with 3 classes 
```bash
python audio_classification.py -c 3
```


### Audio regression
Regression using Pytorch and Librosa

Run instructions for regression
```bash
python audio_regression.py -c <class>
```

Regression on width 
```bash
python audio_regression.py -c width
```
