# Model: SSED - Sliding-window Sound Event Detection with width and height.

Source: [GitHub-link](https://github.com/bendikbo/SSED).

Master Thesis: [Thesis](https://github.com/bendikbo/SSED/blob/main/thesis.pdf).

Status: Runs, but poor results. 

## About
This is the same SSED-repository for Sound Evenet Detection Classification. But modified to train on the width and height. Thus, there has been some changes in the dataloaders, and in classifers/models. The neural networks are forwarding both a probablity-distribution and the width and height. The model must be trained for several hours and will print the targets and predictions for each validation epoch. The results are not that great, but the model might be close to be finished. If you want to develop the model further; it is advised to experiment with the model-architecture and loss-functions in classifier/models. Beware you need a lot of compute.

# Troubleshooting
If you receive som CUDA-error, please do some Google'ing regarding you error or reduce the batchsize.

# References
<a id="1">[1]</a> 
Bendik Bogfjellmo (2021). 
Two deep learning approaches to Sound Event Detection for bird sounds in the arctic biosphere
Master’s thesis in Electronics Systems Design and Innovation NTNU Supervisor: Guillaume Dutilleux July 2021

