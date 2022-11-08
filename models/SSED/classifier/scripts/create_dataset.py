#!/usr/bin/env python3
"""
Label parsing script written to make self annotated bird song dataset into
more pytorch friendly format, will probably have to make some alterations
for this script to make it applicable for users. But should be able to 
give a good start to create this functionality.

TODO: Fix this piece of specialized garbage.
"""
import argparse
import os
import numpy as np
import pandas as pd
from math import ceil, floor
from pydub import AudioSegment
from classifier.config.defaults import cfg
from classifier.data.datasets import dereference_dict



def parse_labels(label_file_path: str) -> dict:
    """
    Function to parse labels
    Input:
    label_file_path - path to audacity-style label file
    Ouput:
    dict in style of label_dict["class_name"]=(onset, offset, low_freq, high_freq)
    """
    assert os.path.exists(label_file_path), f"Could not find label file: {label_file_path}\n"
    label_file = open(label_file_path)
    label_lines = label_file.readlines()
    label_file.close()
    label_dict = {}
    for line_no, label_line in enumerate(label_lines):
        #backslash implies it's a frequency range line
        if label_line[0] == "\\":
            continue
        label_vals = label_line.rstrip("\n").split("	")
        #get class of annotation
        print(label_vals)
        voc_class = label_vals[2]
        #don't need it in tuple
        del label_vals[2]
        label_vals.extend(label_lines[line_no + 1].rstrip("\n")[2:].split("	"))
        #Turn values into floats
        label_vals = [float(val) for val in label_vals]
        #Check if dict already contain annotation of this class
        if voc_class in label_dict:
            label_dict[voc_class].append(tuple(label_vals))
        else:
            label_dict[voc_class] = [tuple(label_vals)]
    return label_dict


def split_to_dirs(
    label_dict: dict,
    wav_file_path: str,
    buffer = 0.0,
    output_path = "output/"
    ):
    """
    Function to create separate folders for each class
    and store audio of each annotation as a separate file
    Inputs:
    label_dict - dict in style of label_dict["class_name"][index]=(onset, offset, low_freq, high_freq)
    wav_file_path - path to wav file
    buffer = 0.0 - buffer added before and after annotation in input file to create output audio segment
    """
    if output_path != "":
        output_path = output_path + "/"
        os.system(f"rm -rf {output_path}")
        os.system(f"mkdir {output_path}")
    #Verify that wav path exists
    assert os.path.exists(wav_file_path), f"Could not find input wav file: {wav_file_path}\n"
    record = AudioSegment.from_wav(wav_file_path)
    for voc_class in label_dict.keys():
        if os.path.exists(output_path + voc_class):
            #Dirty solution to clean directory
            os.system("rm -rf " + output_path + " " + voc_class)
        os.system(f"mkdir {output_path}{voc_class}")
        for voc_no, annotation in enumerate(label_dict[voc_class]):
            #Assuming onset and offset is measured in seconds
            onset, offset, _, _ = annotation
            if onset < buffer:
                onset = 0
            #Recalculating onset and offset to comply with pydub
            if onset < buffer:
                onset = 0
            else:
                onset = int(floor((onset-buffer)*1000))
            if (offset + buffer) > (len(record)/1000):
                offset = len(record) - 1
            else:
                offset = int(floor((offset+buffer)*1000))
            vocalization = record[onset:offset]
            vocalization.export(f"{output_path}{voc_class}/{voc_no}.wav", format="wav")


def compound_class_record(
    label_dict: dict,
    wav_file_path: str,
    buffer = 1.0,
    make_label_file = False,
    output_path = "output/",
    crossfade = True
    ):
    """
    Function to create compound wav file for all occurences for each class
    Inputs:
    label_dict - dict in style of label_dict["class_name"][index]=(onset, offset, low_freq, high_freq)
    wav_file_path - path to wav file
    buffer = 0.0 - buffer added before and after 
    annotation in input file to create output audio segment
    """
    if output_path != "":
        os.system(f"rm -rf {output_path}")
        os.system(f"mkdir {output_path}")
    assert os.path.exists(wav_file_path), f"Could not find input wav file: {wav_file_path}\n"
    record = AudioSegment.from_wav(wav_file_path)
    for voc_class in label_dict.keys():
        os.system(f"mkdir {output_path}{voc_class}")
        compound_sound = record[0]
        output_labels = ""
        for annotation in label_dict[voc_class]:
            onset, offset, low_freq, high_freq = annotation
            #Recalculating onset and offset to comply with pydub
            if onset < buffer:
                onset = 0
            else:
                onset = int(floor((onset-buffer)*1000))
            if (offset + buffer) > (len(record)/1000):
                offset = len(record) - 1
            else:
                offset = int(floor((offset+buffer)*1000))
            if make_label_file:
                comp_onset = float(len(compound_sound)/1000) + buffer
                comp_offset = float(comp_onset) + float((offset-onset)/1000) - buffer*2
                if crossfade:
                    comp_onset -= min(len(compound_sound)*1000, buffer/2)
                    comp_offset -= min(len(compound_sound)*1000, buffer/2)
                comp_onset = str(comp_onset)
                comp_offset = str(comp_offset)
                low_freq = str(low_freq)
                high_freq = str(high_freq)
                output_labels += \
                "{comp_onset}	{comp_offset}	" + \
                "{voc_class+str(timedelta(seconds=onset//1000))}\n\\"+\
                "{low_freq}	{high_freq}\n"
            vocalization = record[onset:offset]
            compound_sound = compound_sound.append(vocalization, crossfade = min((buffer*1000)//2, len(compound_sound)))
        compound_sound.export(f"{output_path}{voc_class}/{voc_class}.wav", format = "wav")
        if make_label_file:
            if os.path.exists(f"{output_path}{voc_class}.txt"):
                os.remove(f"{output_path}{voc_class}.txt")
            label_file = open(f"{output_path}{voc_class}/{voc_class}.txt", 'a')
            label_file.write(output_labels)
            label_file.close()


def create_dataset(annotations_dir_path : str, valid_classes : list, val_amount = 0.17):
    """
    Creates test/val dataset from directory containing directories of .wav-files and .txt-audacity-label-files
    The .wav-files and the .txt-audacity-label-files must be the same name as the directory they're in
    For each "BEGIN" and "END" in the label-file, creates a wav file with it's own csv file
    containing the annotations from the .txt-audacity label file.
    Input:
    annotations_dir_path - relative path to the directory containing directories of .wav/.txt-annotation combos
    valid_classes - list of strings containing the name of the valid classes for the detector
    val_amount - amount of validation data in test/val split*

    *time-frames are selected for validation by pseudo-random chance
    """
    #purge/create directories for wav/csv-files
    os.system("rm -rf train \n mkdir train \n rm -rf val \n mkdir val")
    #Annotations_dir_path contains directories with .wav and .txt label files
    annotation_dirs_paths = os.listdir(annotations_dir_path)
    for annotation_path in annotation_dirs_paths:
        wav_path = f"{annotations_dir_path}/{annotation_path}/{annotation_path}.wav"
        label_path = f"{annotations_dir_path}/{annotation_path}/{annotation_path}.txt"
        label_dict = parse_labels(label_file_path=label_path)
        record = AudioSegment.from_wav(wav_path)
        valid_classes.extend(["BEGIN", "END"])
        #Check if an end or a beginning has been forgotten
        if len(label_dict["END"]) != len(label_dict["BEGIN"]):
            print(f"mismatch of timeframe BEGIN and END in {label_path}")
        #Check for erronous labels
        elif not all(voc_class in valid_classes for voc_class in label_dict):
            print(f"{label_path} contains invalid annotation")
            print("labels in file:")
            for voc_class in label_dict:
                print(voc_class)
        else:
            for time_frame_no, beginning in enumerate(label_dict["BEGIN"]):
                begin_time = beginning[0]
                end_time = label_dict["END"][time_frame_no][0]
                assert (end_time-begin_time) > 60, f"time frame in: {label_path} with less than 60 seconds, starts at: {begin_time}"
                if (end_time - begin_time) > 200:
                    print(f">200 second time frame in {label_path} starting at {begin_time}")
                #rows in csv output
                rows = []
                for valid_label in valid_classes:
                    #Create list of lists containing [label, onset, offset] for rows in csv file
                    if valid_label not in label_dict.keys():
                        continue
                    rows.extend([
                        [valid_label,
                        voc[0]-begin_time, voc[1]-begin_time]
                        for voc in label_dict[valid_label] if (voc[0] > begin_time and voc[1] < end_time)
                        ])
                time_frame_record = record[int(floor(begin_time*1000))]
                time_frame_record += record[int(floor(begin_time*1000))+1 : int(ceil(end_time*1000))]
                df = pd.DataFrame(np.array(rows), columns=["class", "onset", "offset"])
                if np.random.uniform() < val_amount:
                    df.to_csv(f"val/{annotation_path}_{time_frame_no}.csv")
                    time_frame_record.export(out_f=f"val/{annotation_path}_{time_frame_no}.wav", format="wav")
                else:
                    df.to_csv(f"train/{annotation_path}_{time_frame_no}.csv")
                    time_frame_record.export(out_f=f"train/{annotation_path}_{time_frame_no}.wav", format="wav")
                


def main(cfg):
    deref_dict = dereference_dict(cfg.INPUT.NAME)
    class_name_list = [value for value in deref_dict.values()]
    print(class_name_list)


def get_parser():
    parser = argparse.ArgumentParser(
        description='Sound Event Detection dataset creation')
    parser.add_argument(
        "config_file",
        default="",
        metavar="config_file",
        help="path to config file",
        type=str,
    )
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    cfg.merge_from_file(args.config_file)
    main(cfg)

