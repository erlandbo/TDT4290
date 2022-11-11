# Model: SSED - Sliding-window Sound Event Detection

Source: [GitHub-link](https://github.com/bendikbo/SSED).

Master Thesis: [Thesis](https://github.com/bendikbo/SSED/blob/main/thesis.pdf).

Status: Runs.

## About

The SSED repository contains a sliding window sound event detection system developed for a master thesis at NTNU IES by Bendik Bogfjellmo. Sound Event Detection systems should utilize a deep learning model to produce an output, formatted as a sound event label, the starting time of the sound event (onset), and the end time of the sound event (offset) [1]. The models will try to classify vehciles by three classes: small, medium and large. 

It is highly recommended to read the master thesis and the original repo for further details.

# Changes
We have made very few changes to the original repo. 

- Build Dataset script: The original ssed-repo expects the dataset to be manually labeled in Audacity [1]. Since the group were already provided with a labeled dataset and did not have enough time to manually re-label the dataset, we decided to build a script for building the dataset. The script SSED/build_dataset.py will iterate over each 10seconds interval in the audio and build a dataframe by vehicles which are active in the 10second interval. The dataset will use a Sample Rate of 16kHz. Checkout SSED/build_dataset.py for details.
- Configuration: Changed the number of class from 5 to 3 in SSED/configs/defaults.yaml
- Configuration: Changed the number of classes from 5 to 3 in SSED/classifier/config/defaults.py 
- Configuration: Changed the train-batch size from 32 to 32//4 = 8. We used a division so we would remember the original-batch size
- Configuration: Changed the test-batch size from 32 to 32//4 = 8. 
- Configuration: Changed the dereference kauto_dict dictionary from bird-species to vehicle classes in SSED/classifier/data/dataset/__init__py 
- Configuration: Changed the dictionary label_dict from bird-species to vehicle classes in SSED/classifier/data/dataset/kauto5cls.py


# Hardware requirements
The model has been tested on these Linux systems 
- Nvidia RTX 3060 12GB VRAM, Intel Core i9-10850K, 32GB RAM, Ubuntu 22.04
- NVidia Tesla T4, 16 vCPU, 32 GB RAM, Debian 10, Google Cloud

Try to reduce the batch-size if you are unable to run the model with the default specifications.

# Configurations
If you want to change the configurations checkout SSED/classifiser/config/defaults.py and SSED/configs/default.yaml
We have set the number of classes to 3; namely small, medium and large vehicles.

# Results
With the default configurations and hardware requirements, the models should get decent results after approximately 25 epochs.

After some epochs with training the codebase should achieve a high AP (approximately 0.8-0.95 AP) for both medium and large vehilces while the AP for small vehicles varies between 0.1-0.7 depending on the dataset used. The total MAP will thus be affected by the difficulties with small vehicles and thus be somewhere between 0.6-0.9 depending on the dataset and difficulties with small vehicles. There might be some imbalance in the dataset and resampling by adding more examples of small-vehicles has often given better results for small vehicles. Checkout the master-thesis for details regarding MAP.


# Troubleshooting
If you receive some CUDA-error, try to reduce the batchsize.


# References
<a id="1">[1]</a> 
Bendik Bogfjellmo (2021). 
Two deep learning approaches to Sound Event Detection for bird sounds in the arctic biosphere
Master’s thesis in Electronics Systems Design and Innovation NTNU Supervisor: Guillaume Dutilleux July 2021

