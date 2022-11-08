import torch
import torchaudio
import numpy as np
import os
from classifier.data.transform.transforms import AudioTransformer


class WindowSlide(torch.utils.data.Dataset):

    def __init__(
                self,
                cfg,
                audio_file:str
                ):
        assert os.path.exists(audio_file)
        self.input_length = cfg.INPUT.RECORD_LENGTH
        hops_per_window = getattr(cfg.INFERENCE, "HOPS_PER_WINDOW", 4)
        self.hop_size = self.input_length // hops_per_window
        self.transform = AudioTransformer(cfg.INPUT.TRANSFORM, is_train=False)
        self.sample_rate = cfg.INPUT.SAMPLE_FREQ
        self.record, fs = torchaudio.load(audio_file)
        #In case of stereo or not correct sample rate, resample to fit model
        if fs != self.sample_rate:
            self.resampler = torchaudio.transforms.Resample(
                orig_freq=fs,
                new_freq=self.sample_rate
            )
            self.record = self.resampler(self.record)
        if self.record.size()[0] != 1:
            self.record = torch.mean(self.record, dim=0).unsqueeze(0)
    def __getitem__(self, idx):
        audio_bit_start = min(
            self.record.size()[1] - self.input_length,
            idx * self.hop_size
        )
        audio_bit = torch.narrow(
            self.record,
            1,
            audio_bit_start,
            self.input_length
        )
        audio_bit, _, _ = self.transform(audio_bit)
        return audio_bit, idx

    def __len__(self):
        return int(np.ceil(self.record.size()[1] / self.hop_size))

