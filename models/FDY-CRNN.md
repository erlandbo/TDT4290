# Model: FDY-CRNN

Source: [GitHub-link](https://github.com/frednam93/FDY-SED).

Status: Unable to run due to technical difficulties.

## About

FDY-CRNN is currently a state of the art model on sound event detection on the [DESED](https://github.com/turpaultn/DESED) dataset (source: [paperswithcode](https://paperswithcode.com/task/sound-event-detection)). 

The DESED dataset is a dataset of a lot of short audio clips and their corresponding metadata based on the type of dataset it is. The dataset contains a real dataset, a synthetic dataset, and a public evaluation dataset. The real dataset, being a strongly labeled dataset with a folder of short audio files, and a csv file containing the a short time frame (onset and offset) in a file and the label of that time frame, being what it is the sound of. The classes of the dataset are the following: Alarm/bell/ringing, Blender, Cat, Dog, Dishes,
Electric shaver/toothbrush, Frying, Running water, Speech, Vacuum cleaner.

The team hoped that this model would be a good classification model for our dataset as it classified based on the recognition of sound. However, the model did not classify sound based on small nuance differences, but rather quite different sounds, which could affect the accuracy of the model on the team's sound data of passing vehicles. Still, since the model is a state of the art model on sound event detection, the team thought it was worth a try. 

## The difficulties with the model
Firstly, the team tried to make the model run on the DESED dataset. After several downloads of small variations of the dataset from different channels ([DESED GitHub](https://github.com/turpaultn/DESED), [DCASE 2021 Task 4 description page](http://dcase.community/challenge2021/task-sound-event-detection-and-separation-in-domestic-environments) and [DCASE 2021 Task 4 baseline](https://github.com/DCASE-REPO/DESED_task)), the team was not able to run the model on the original dataset. This was because each of the downloaded datasets was missing the corresponding durations file to each of the datasets that this FDY-CRNN model used. Thus, the team was unable to run the model on its original dataset. 

The missing corresponding durations files to each of datasets also meant that the team was unable to see the structure of these files, and thus struggled to create them when adapting the model to their own dataset. In addition, the DESED dataset has multiple datasets that the FDY-CRNN model trains on. Changing the model to only run using a strongly labeled dataset could impact the results. Therefore, the team was unable to adapt the model to the vehicle dataset. 

The team also tried to use the FDY-CRNN model itself within the SSED setup. However, the size of the input to the "forward()" method of CRNN did not match the input size that the setup of SSED put in. The CRNN model expected a tensor with size \[batch_size, frequencies, frames\], while the SSED setup put in a tensor with size torch.Size(\[8, 3, 224, 224\]). 

## Conclusion
The team was unable to run this model, and thus cannot say whether this model is good or bad for classifying vehicles on the road. This was because changing the model to work on the vehicle sound dataset required considerable changes to both the setup of the model and the vehicle sound dataset. Because of the time constraints of this project, and because the team was unsure whether this model would actually produce good results after these modifications, the team decided not to continue work on this model. During the course of the project, the team was asked to prefer estimating the dimensions of the vehicles over classification. Because of the complexity of the FDY-CRNN model, the team also deemed it difficult to change the model to perform regression to estimate dimensions instead of classification. It might still be interesting to look at the FDY-CRNN model if more time, resources and a knowledgeable team are put to continue this project. However, for this project the team decided it was better to focus on other models instead. 
