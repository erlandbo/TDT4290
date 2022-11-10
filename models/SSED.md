# Model: SSED - Sliding-window Sound Event Detection

Source: [GitHub-link](https://github.com/bendikbo/SSED).

Master Thesis: [Thesis](https://github.com/bendikbo/SSED/blob/main/thesis.pdf).

Status: Runs.

## About

The SSED repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES by Bendik Bogfjellmo, with contributions by NINA - Norsk Institutt for NAturfroskning. The problem of detection and classification of event vocalizations falls into a broader category of machine learning problems commonly referred to as Sound Event Detection (SED). The system should utilize a deep learning model to produce the output. The outputs should be formatted as a sound event label, the starting time of the sound event (onset), and the end time of the sound event (offset) [1].

It is highly recommended to read the master thesis and the original repo for further details.


# Models
The model used for classification can be set in SSED/classifiser/config/defaults.py and SSED/configs/default.yaml. The codebase includes EfficientNet, ResNet34 and ResNet50. EfficientNet is set by default.

# Dataset

The original author of the ssed-repo used a self-described active window"-method in Audacity. The method allows to only annotate parts of an audio file, as long as you label the start of an annotated section of the file with "BEGIN" and the end of the annotated section with "END". So all the data had to be manually labeled in Audacity [1]. Since the group were already provided with labeled datasets and did not have  the time to manually re-label all the events, the group decided to build a script for splitting the audio. The script splits the audio in 10-seconds intervals with a corresponding .csv. The script is available in SSED/build_dataset.py.

# Hardware requirements
The model has been tested on these Linux systems 
- Nvidia RTX 3060 12GB VRAM, Intel Core i9-10850K, 32GB RAM, Ubuntu 22.04
- NVidia Tesla T4, 16 vCPU, 32 GB RAM, Debian 10, Google Cloud

But you might run the model on lower compute power. Try to reduce the batch-size if you are unable to run the model with the default specifications.

# Configurations
We have set the number of classes to 3; namely small, medium and large vehicles.

Allmost all of the configurations from the bird-classification are kept, except for decreasing the batchsize from 32 to 8 for training and testing. Checkout the the configurations in SSED/classifiser/config/defaults.py and SSED/configs/default.yaml and the master thesis for more details. If you want to change the classification; take a look at classifier/data/datasets/__init__.py and classifier/data/datasets/kauto5cls.py.
As mentioned in the master thesis, the results are very dependent on the windows-size, number of hops per windows and IoU-treshhold. it is encouraged to experiment with these to improve the results.  But you may want the run more epochs and increase/decrease the batchsize depending on the results. A samling rate of 16kHZ has been set on all audio data

# Results
With the default configurations and hardware requirements, the models should get decent results after approximately 25 epochs. The master thesis used 5740 sound events annotated from 450 hours of audio [1]. This is significantly more than our total amount of hours of recording of vehicles (approximately 11-12 hours and 3000 vehicles).

After some epochs with training the codebase should achieve a high AP (approximately 0.8-0.95 AP) for both medium and large vehilces while the AP for small vehicles varies between 0.1-0.7 depending on the dataset used. The total MAP will thus be affected by the difficulties with small vehicles and thus be somewhere between 0.6-0.9 depending on the dataset and difficulties with small vehicles. There might be some imbalance in the dataset and resampling by adding more examples of small-vehicles has often given better results for small vehicles. Checkout the master-thesis for details regarding MAP.


# Troubleshooting
If you receive some CUDA-error, please do some Google'ing regarding you error or reduce the batchsize.


# References
<a id="1">[1]</a> 
Bendik Bogfjellmo (2021). 
Two deep learning approaches to Sound Event Detection for bird sounds in the arctic biosphere
Master’s thesis in Electronics Systems Design and Innovation NTNU Supervisor: Guillaume Dutilleux July 2021

