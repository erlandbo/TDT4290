"""
Dataset class for self annotated Kautokeino bird vocalization dataset
Written (entirely) by:
Bendik Bogfjellmo (github.com/bendikbo) (bendik.bogfjellmo@gmail.com)
"""
import torch
import torchaudio
import numpy as np
import os
import pandas as pd


class Kauto5Cls(torch.utils.data.Dataset):

    def __init__(
                self,
                data_dir:str,
                transform=None,
                target_transform=None,
                is_train=True,
                inference_filepath="",
                audio_bit_length=0.0
                ):
        self.is_train = is_train
        if inference_filepath != "":
            self.inference = True
            self.entire_record, sample_rate = torchaudio.load(inference_filepath)
            self.length = self.entire_record.size()[1] / sample_rate
            #Can't have test and inference mode at the same time.
            self.is_train = False
        else:
            self.inference = False
        self.label_dict = {
                        "True"   :   0,
                        #"bekkasinflukt"     :   1,
                        #"småspove-sang"     :   2,
                        #"heilo-pip"         :   3,
                        #"heilo-sang"        :   4
                    }
        filenames = sorted(os.listdir(data_dir))
        self.data_dir = data_dir
        self.audio_filenames = []
        #audio_bits variable is used for validation to split up larger audio files
        self.audio_bits = []
        self.audio_bit_length = audio_bit_length
        for filename in filenames:
            if ".csv" not in (filename.lower()):
                self.audio_filenames.append(filename)
        self.transform = transform
        self.target_transform = target_transform
        self.annotation_dict = {}
        for audio_filename in self.audio_filenames:
            #csv should have same name as audio file except extension
            csv_filename = audio_filename.split(".")[0] + ".csv"
            #Using C engine because it's supposed to be faster, requires delimeter to be commas
            data_frame = pd.read_csv(data_dir + "/" + csv_filename, engine="c")
            #Storing all annotations in dictionary
            self.annotation_dict[audio_filename]=[]
            for _, row in data_frame.iterrows():
                self.annotation_dict[audio_filename].append(
                        (float(row["onset"]),
                        float(row["offset"]),
                        str(row["class"]),
                        int(row["width"]),
                        int(row["height"]))
                    )
        if not self.is_train:
            #Have to create a data loading method to create validation dataset
            self.convert_to_validation()
        self.validate_dataset()

    def convert_to_validation(self):
        """
        Function to create dataset for validation and/or test
        fills self.audio_bits with filenames and start time which
        is later used to crop the audio files through a transform.
        The cropping is done so that the audio bits always have 50%
        overlap with the next sequence, as this is how inference is
        intended to work on the dataset.
        """
        for audio_filename in self.audio_filenames:
            annotations = self.annotation_dict[audio_filename]
            earliest = min(annotations, key=lambda t:t[0])[0]
            latest = max(annotations, key=lambda t:t[1])[1]
            if earliest < 1:
                earliest = 0
            else:
                earliest = earliest - 1.0
            while (earliest + self.audio_bit_length) < latest:
                #Audio bits contain list of (filename, start(sec))
                self.audio_bits.append(
                    (audio_filename,
                    earliest)
                    )
                earliest += (self.audio_bit_length / 2)
            #No need to add latest annotation if it's over before 32 secs
            if latest > self.audio_bit_length:
                self.audio_bits.append(
                    (audio_filename,
                    latest - self.audio_bit_length)
                )
        
    def validate_dataset(self):
        for audio_filename in self.audio_filenames:
            assert audio_filename in self.annotation_dict,\
                f"Did not find label for record {audio_filename} in labels"

    def __getitem__(self, idx):
        if self.is_train:
            lines, labels = self.get_annotation(idx)
            record = self._read_record(idx)
        #Don't have to care about lines or labels during inference
        elif self.inference:
            start_sec = idx * self.audio_bit_length / 2
            if (start_sec + self.audio_bit_length) > self.length:
                start_sec = self.length - self.audio_bit_length
            record, _, _ = self.transform.validation_crop(
                self.entire_record,
                start_sec,
                lines = None,
                labels = None
            )
            timespan = (start_sec, start_sec + self.audio_bit_length)
            return record, timespan, idx
        #Have to care about lines and labels during validation/testing
        else:
            audio_filename, start_sec = self.audio_bits[idx]
            lines, labels = self._get_annotation(audio_filename)
            record, _ = torchaudio.load(self.data_dir + "/" + audio_filename)
            #ValCrop object initalized in transform when is_train is set false
            record, lines, labels = self.transform.validation_crop(
                                                    record,
                                                    start_sec,
                                                    lines,
                                                    labels
                                                )
        if self.transform:
            record, lines, labels = self.transform(record, lines, labels)
        if self.target_transform is not None:
            targets = self.target_transform(lines, labels)
        return record, targets, idx

    def __len__(self):
        if self.is_train:
            return len(self.audio_filenames)
        else:
            return len(self.audio_bits)

    def _get_annotation(self, audio_filename):
        annotations = self.annotation_dict[audio_filename]
        lines = np.zeros((len(annotations), 2), dtype=np.float32)
        labels = np.zeros((len(annotations), 3), dtype=np.int64)
        for idx, annotation in enumerate(annotations):
            line = [
                annotation[0],
                annotation[1]
            ]
            lines[idx] = line
            labels[idx] = self.label_dict[annotation[2]], annotation[3], annotation[4]
        return lines, labels

    def get_annotation(self, index):
        audio_filename = self.audio_filenames[index]
        return self._get_annotation(audio_filename)

    def _read_record(self, idx):
        audio_filename = self.audio_filenames[idx]
        audio_filepath = self.data_dir + "/" + audio_filename
        record, _ = torchaudio.load(audio_filepath)
        return record
