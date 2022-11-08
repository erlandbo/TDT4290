# Feature exploration
 
By performing feature exploration we were able to detect several important features that could be used in classefication and regression task. 





## Audio features

| Feature            | Model                | Description                                                                                                                                                                                     |
|--------------------|----------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| MFCC               | forest/decision tree | mel-frequency cepstrum (MFC) is a representation of the short-term power spectrum of a sound, based on a linear cosine transform of a log power spectrum on a nonlinear mel scale of frequency. |
| MEL spectrogram    | RESNET/SSD           | A spectrogram where the frequencies are converted to the mel scale.                                                                                                                             |
| Zero crossing rate | forest/decision tree | Number of times the audio signal crosses 0.                                                                                                                                                     |
| RMS energy         | Vehicle detection    | Root mean square of the total magnitude of the signal. Correnspond to how loud the signal is.                                                                                                   |
| Spectral Centroid  | Not used             | Indicates at which frequency the energy of a spectrum is centered upon.                                                                                                                         |
