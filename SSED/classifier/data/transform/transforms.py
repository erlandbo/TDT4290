import torch
from torch import Tensor
import torch.nn as nn
import numpy as np
import torchvision
import torchaudio


class ToSampleCoords(nn.Module):
    """
    Pytorch module to convert coordinates measured in seconds
    into coordinates measured in sample Nos,
    Default sample rate is 16kHz unless this is set through cfg.SAMPLE_RATE
    """
    def __init__(self, cfg):
        super(ToSampleCoords, self).__init__()
        self.sample_rate = 16000
        if hasattr(cfg, "SAMPLE_RATE"):
            self.sample_rate = cfg.SAMPLE_RATE
    def forward(self, x, lines=None, labels=None):
        if lines is not None:
            for idx, annotation in enumerate(lines):
                onset = annotation[0]
                offset = annotation[1]
                new_onset = np.ceil(onset*self.sample_rate)
                new_offset = np.floor(offset*self.sample_rate)
                lines[idx] = [new_onset, new_offset]
        return x, lines, labels


class Differentiate1D(nn.Module):
    """
    Pytorch module to discretely differentiate a 1D input tensor
    Differentiates by taking tensor[1:end] - tensor[0:end-1]
    """
    def __init__(self, cfg):
        super(Differentiate1D, self).__init__()
        if hasattr(cfg.DIFFERENTIATE, "STEP"):
            self.step = cfg.DIFFERENTIATE.STEP
        else:
            self.step = 1
        
    def forward(self, x, lines = None, labels = None):
        minuend = torch.narrow(x, 0, self.step, x.size()[0] - self.step)
        subtrahend = torch.narrow(x, 0, 0, x.size()[0] - self.step)
        x = minuend - subtrahend
        return x, lines, labels


class RandFlip1D(nn.Module):
    """
    Pytorch module to "flip" signal along the x-axis
    Supports random application through cfg.RAND_FLIP.CHANCE or cfg.CHANCE
    defaults to random application with p = 0.5
    """
    def __init__(self, cfg):
        super(RandFlip1D, self).__init__()
        #Code block for setting up random application
        if hasattr(cfg.RAND_FLIP, "CHANCE"):
            self.chance = cfg.RAND_FLIP.CHANCE
        elif hasattr(cfg, "CHANCE"):
            self.chance = cfg.CHANCE
        else:
            self.chance = 0.5
    def forward(self, x : Tensor, lines = None, labels = None):
        if np.random.uniform() < self.chance:
            x = x.mul(-1)
        return x, lines, labels


class RandGauss1D(nn.Module):
    """
    Pytorch module to add gaussian noise to 1D tensor
    Supports random application through cfg.RAND_FLIP.CHANCE or cfg.CHANCE
    Also supports noise intensity through INTENSITY
    For uniformly distributed intensity, include a RAND to cfg.GAUSS
    """
    def __init__(self, cfg):
        super(RandGauss1D, self).__init__()
        #Code block for random application
        if hasattr(cfg.RAND_GAUSS, "CHANCE"):
            self.chance = cfg.RAND_GAUSS.CHANCE
        elif hasattr(cfg, "CHANCE"):
            self.chance = cfg.CHANCE
        else:
            self.chance = 0.5
        self.intensity = 1.0
        if hasattr(cfg.RAND_GAUSS, "INTENSITY"):
            self.intensity = cfg.RAND_GAUSS.INTENSITY
        self.random_intensity = False
        if hasattr(cfg.RAND_GAUSS, "RAND"):
            self.random_intensity = True
    def forward(self, x : Tensor, lines = None, labels = None):
        if np.random.uniform() < self.chance:
            #noise_factor = std(x) * intensity
            noise_factor = x.std() * self.intensity
            if self.random_intensity:
                noise_factor *= float(np.random.uniform())
            #x_i + N(0,1) * noise_factor
            x += torch.randn(x.size()) * noise_factor
        return x, lines, labels


class RandAmpAtt1D(nn.Module):
    """
    Pytorch module to randomly amplify or attenuate signal
    Supports random application through cfg.RAND_AMP_ATT.CHANCE or cfg.CHANCE
    defaults to random application with p = 0.5
    cfg.AMP_ATTEN is required to have parameter "FACTOR"
    """
    def __init__(self, cfg):
        super(RandAmpAtt1D, self).__init__()
        #Code block for setting up random application
        if hasattr(cfg.RAND_AMP_ATT, "CHANCE"):
            self.chance = cfg.RAND_AMP_ATT.CHANCE
        elif hasattr(cfg, "CHANCE"):
            self.chance = cfg.CHANCE
        else:
            self.chance = 0.5
        assert hasattr(cfg.RAND_AMP_ATT, "FACTOR"),\
            "Transform AmpAtt1D requires parameter cfg.FACTOR"
        self.factor = max(1/cfg.RAND_AMP_ATT.FACTOR, cfg.RAND_AMP_ATT.FACTOR)
    def forward(self, x : Tensor, lines = None, labels = None):
        if np.random.uniform() < self.chance:
            factor = np.random.uniform(low = 1/self.factor, high = self.factor)
            x.mul(factor)
        return x, lines, labels


class RandContrast1D(nn.Module):
    """
    Pytorch module to add random contrast to the data
    Supports random application through cfg.RAND_CONTRAST.CHANCE or cfg.CHANCE
    defaults to random application with p = 0.5
    Contrast enhancement amount may range between 0-100
    enhancement of 0 still yields a significant contrast enhancement
    """
    def __init__(self, cfg):
        super(RandContrast1D, self).__init__()
        #Code block for setting up random application
        if hasattr(cfg.RAND_CONTRAST, "CHANCE"):
            self.chance = cfg.RAND_CONTRAST.CHANCE
        elif hasattr(cfg, "CHANCE"):
            self.chance = cfg.CHANCE
        else:
            self.chance = 0.5
        assert hasattr(cfg.RAND_CONTRAST, "ENHANCE"), "RandContrast1D needs attribute cfg.RAND_CONTRAST.ENHANCE)"
        self.enhancement = cfg.RAND_CONTRAST.ENHANCE
        if self.enhancement < 0 or self.enhancement > 100:
            print(f"enhancement not in 0-100 range, setting to {np.abs(self.enhancemnet % 100)}")
            self.enhancement = np.abs(self.enhancement % 100)
    def forward(self, x : Tensor, lines = None, labels = None):
        if np.random.uniform() < self.chance:
            amount = np.random.uniform()*self.enhancement
            torchaudio.functional.contrast(waveform=x, enhancement_amount=amount)
        return x, lines, labels


class Crop1D(nn.Module):
    """
    Pytorch module to crop one dimensional signal
    Supports "random" mode and "center" mode  through cfg.CROP.TYPE
    defaults to random application with p = 1.0 unless cfg.CROP.CHANCE is set
    If the chance doesn't activate, it defaults to crop
    to the middle of the input time series.
    Onset/offset annotations can optionally be added through the lines variable
    """
    def __init__(self, cfg):
        super(Crop1D, self).__init__()
        #Code block for setting random application
        self.type = "random"
        if hasattr(cfg.CROP, "TYPE"):
            self.type = cfg.CROP.TYPE
        if hasattr(cfg.CROP, "CHANCE"):
            self.chance = cfg.CROP.CHANCE
        else:
            self.chance = 1
        #Chance set to 0 => centercrop
        if self.type == "center":
            self.chance = 0
        assert hasattr(cfg, "LENGTH"), "Crop1D needs output tensor length to function"
        self.length = cfg.LENGTH
    def forward(self, x : Tensor, lines=None, labels=None):
        start = int(np.floor((x.size()[1]-self.length)/2))
        if np.random.uniform() < self.chance:
            min_start, max_start = (0, x.size()[1] - self.length - 1)
            start = np.random.randint(low=min_start, high = max_start)  
        x = x.narrow(1, start, self.length)
        #Onset/onset annotations need to be fixed if they're added
        if lines is not None:
            new_lines = []
            new_labels = []
            for idx, annotation in enumerate(lines):
                #Have to deduct starting point from the onset
                annotation_onset = annotation[0] - start
                #End = starting point + annotation length
                annotation_offset = annotation_onset + (annotation[1] - annotation[0])
                #Fix out of bounds issues
                if annotation_offset > self.length:
                    annotation_offset = self.length
                if annotation_onset < 0:
                    annotation_onset = 0
                if annotation_onset > self.length or annotation_offset < 0:
                    continue
                else:
                    new_lines.append([annotation_onset, annotation_offset])
                    new_labels.append(labels[idx])
            lines = np.zeros((len(new_lines), 2), dtype=np.float32)
            labels = np.zeros((len(new_labels)), dtype=np.int64)
            for idx, new_line in enumerate(new_lines):
                lines[idx] = new_line
                labels[idx] = new_labels[idx]
        return x, lines, labels


class Spectrify(nn.Module):
    """
    Pytorch module to convert 1D tensor to spectrogram(s)
    cfg.RESOLUTION to specify [WIDTH, HEIGHT]
    cfg.CHANNELS to specify channel order between ["mel", "log", "normal"]
    cfg.FREQ_CROP, if only part of the spectrogram frequency dimension is needed
    All outputs are normalized
    """
    def __init__(self, cfg):
        super(Spectrify, self).__init__()
        self.length = cfg.LENGTH
        #Setting imagenet resolution as defaults
        self.width = 224
        self.height = 224
        sample_freq = 16000
        if hasattr(cfg.SPECTROGRAM, "RESOLUTION"):
            self.width = cfg.SPECTROGRAM.RESOLUTION[0]
            self.height = cfg.SPECTROGRAM.RESOLUTION[1]
        self.out_width = self.width
        self.out_height = self.height
        if hasattr(cfg.SPECTROGRAM, "FREQ_CROP"):
            self.crop = cfg.SPECTROGRAM.FREQ_CROP
            self.out_height = self.crop[1]
        else:
            self.crop = None
        if hasattr(cfg, "SAMPLE_RATE"):
            sample_freq = cfg.SAMPLE_RATE
        self.transformations = nn.ModuleList()
        self.channels = ["normal","log","mel"]
        self.resize = torchvision.transforms.Resize((self.out_height, self.out_width))
        if hasattr(cfg.SPECTROGRAM, "CHANNELS"):
            self.channels = cfg.SPECTROGRAM.CHANNELS
        for channel in self.channels:
            if channel == "mel":
                #Not sure whether this is right or not
                out_height = self.height
                if self.crop is not None:
                    out_height = self.crop[1]
                hop_size = cfg.LENGTH // self.width
                melify = nn.Sequential(
                    torchaudio.transforms.MelSpectrogram(
                                                n_mels=self.height,
                                                hop_length=hop_size
                        )
                )
                self.transformations.append(melify)
            if channel == "log" or channel == "normal":
                    num_ffts = (self.height - 1)*2 + 1
                    #A bit unsure of line below, but think it should give right width
                    hop_size = cfg.LENGTH // self.width
                    self.transformations.append(
                        torchaudio.transforms.Spectrogram(
                            n_fft = num_ffts,
                            hop_length = hop_size
                        )
                    )
    def forward(self, x : Tensor, lines = None, labels = None):
        if lines is not None:
            new_lines = np.zeros((len(lines), 2), dtype=np.float32)
            for idx, annotation in enumerate(lines):
                #Onset_pixel = onset_sample * spectrogram_width / waveform_sample_length
                annotation_onset = int(np.round((annotation[0]*self.width)/self.length))
                annotation_offset = int(np.round((annotation[1]*self.width)/self.length))
                new_lines[idx] = [annotation_onset, annotation_offset]
            lines = new_lines

        #Convert first element of output to be able to just torch.cat it later
        y = self.transformations[0](x)
        if not (self.crop is None) and self.channels[0] != "mel":
                    y = torch.narrow(
                        input = y,
                        dim = y.dim() - 2,
                        start = self.crop[0],
                        length = self.crop[1]
                    )
        #Normalization after crop
        y = (y - torch.mean(y)) / torch.std(y)
        width = y.size()[-1]
        height = y.size()[-1]
        if width != self.out_width or height != self.out_height:
            y = self.resize(y)
        #If more spectrograms are specified, use them
        if len(self.transformations) > 1:
            for i, transformation in enumerate(self.transformations[1:]):
                channel = transformation(x)
                #No point in cropping mel spectrogram as its done through resize
                #This is because of how the melscale works.
                if not (self.crop is None) and self.channels[i + 1] != "mel":
                    channel = torch.narrow(
                        input = channel,
                        dim = channel.dim() - 2,
                        start = self.crop[0],
                        length = self.crop[1]
                    )
                width = channel.size()[-1]
                height = channel.size()[-2]
                if width != self.out_width or height != self.out_height:
                    channel = self.resize(channel)
                #Statement for logarithmic output.
                if self.channels[i+1] == "log":
                    channel = torch.log(channel)
                #Normalization after crop
                channel = (channel - torch.mean(channel)) / torch.std(channel)
                y = torch.cat((y, channel), 0)
        return y, lines, labels


class ValCrop(nn.Module):
    """
    Pytorch module to crop one dimensional signal specified by
    Supports "random" mode and "center" mode  through cfg.CROP.TYPE
    defaults to random application with p = 1.0 unless cfg.CROP.CHANCE is set
    If the chance doesn't activate, it defaults to crop
    to the middle of the input time series.
    Onset/offset annotations can optionally be added through the lines variable
    """
    def __init__(self, cfg):
        super(ValCrop, self).__init__()
        assert hasattr(cfg, "LENGTH"), "ValCrop needs output tensor length to function"
        self.length = cfg.LENGTH
        self.sample_freq = 16000
        if hasattr(cfg, "SAMPLE_RATE"):
            self.sample_freq = cfg.SAMPLE_RATE
        #If data should be differentiated, a sample has to be added
        if hasattr(cfg, "DIFFERENTATE"):
            self.length += getattr(cfg.DIFFERENTATE, "STEP", default=1)
    def forward(self, x : Tensor, start_second=0.0, lines=None, labels=None):
        signal_length = x.size()[1]
        leftover_samples = signal_length - start_second*self.sample_freq + self.length
        #Implicates that this is the last audio bit in the file
        #In case there is more samples, will then add them to the audio bit
        if leftover_samples < (self.length / 2):
            start_second += min(1.0, leftover_samples/self.sample_freq)
        start = int(np.floor(self.sample_freq * start_second))
        x = x.narrow(1, start, self.length)
        #Onset/onset annotations need to be fixed if they're added
        if lines is not None:
            new_lines = []
            new_labels = []
            for idx, annotation in enumerate(lines):
                #Have to deduct starting point from the onset
                annotation_onset = annotation[0] - start_second
                #End = starting point + annotation length
                annotation_offset = annotation_onset + (annotation[1] - annotation[0])
                #Fix out of bounds issues
                if annotation_offset > self.length/self.sample_freq:
                    annotation_offset = self.length/self.sample_freq
                if annotation_onset < 0:
                    annotation_onset = 0
                if annotation_onset > self.length/self.sample_freq or annotation_offset < 0:
                    continue
                else:
                    new_lines.append([annotation_onset, annotation_offset])
                    new_labels.append(labels[idx])
            lines = np.zeros((len(new_lines), 2), dtype=np.float32)
            labels = np.zeros((len(new_labels)), dtype=np.int64)
            for idx, new_line in enumerate(new_lines):
                lines[idx] = new_line
                labels[idx] = new_labels[idx]
        return x, lines, labels


class AudioTransformer(nn.Module):
    def __init__(self, cfg, is_train=True):
        super(AudioTransformer, self).__init__()
        self.transforms = nn.ModuleList()
        #Implementations of new transforms will have to be added in these
        #dictionaries to be supported by YAML-specification
        if is_train:
            transform_dict = {
                                    "SAMPLE_COORDS" :       ToSampleCoords,
                                    "DIFFERENTIATE" :       Differentiate1D,
                                    "CROP"          :       Crop1D,
                                    "RAND_FLIP"     :       RandFlip1D,
                                    "RAND_GAUSS"    :       RandGauss1D,
                                    "RAND_AMP_ATT"  :       RandAmpAtt1D,
                                    "RAND_CONTRAST" :       RandContrast1D,
                                    "SPECTROGRAM"   :       Spectrify
                             }
        else:
            #Data augmentation and crop is unnecessary on validation/test data
            #As both test and validation data has their own cropping function
            transform_dict = {
                                    "SAMPLE_COORDS" :       ToSampleCoords,
                                    "DIFFERENTIATE" :       Differentiate1D,
                                    "RAND_CONTRAST" :       RandContrast1D,
                                    "SPECTROGRAM"   :       Spectrify
                             }
            self.validation_crop = ValCrop(cfg)
        for transform_name in transform_dict:
            if hasattr(cfg, transform_name):
                if getattr(cfg, transform_name).ACTIVE == True:
                    #This is not allowed with torchscript
                    transform_module = transform_dict[transform_name](cfg)
                    self.transforms.append(transform_module)
    def forward(self, x, lines = None, labels = None):
        for transform in self.transforms:
            x, lines, labels = transform(
                x,
                lines = lines,
                labels = labels
                )
        return x, lines, labels
