import pandas as pd
import numpy as np
import librosa
import soundfile as sf
from math import ceil, floor
from pathlib import Path
import sys
import argparse
import os

def parse_dataframe(datapath, class_):
    df = pd.read_csv(data_path)
    assert class_ in df.columns, "Class not found in dataframe"
    df = df.copy()
    df["entry"] = pd.to_timedelta(df["timedelta_start"]).dt.total_seconds()
    df["exit"] = pd.to_timedelta(df["timedelta_end"]).dt.total_seconds()
    df["class"] = df[class_]
    df = df[["entry", "exit", "class"]]
    return df

def split_to_dirs(datapath, audio_path, class_, name_audio, SR=16000):
    data = parse_dataframe(datapath, class_)
    y, _ = librosa.load(audio_path, sr=SR)
    len_audio_sec = len(y) // SR  # len of audio in sec
    train_num, val_num, test_num = 0, 0, 0
    audio_start, audio_end = 0, 10  # audio segment 10 sec
    buffer = 0.5 ## 1sec buffer
    
    while audio_end < len_audio_sec:
        df = pd.DataFrame(columns=["onset", "offset", "class"])
        active_zone = False
        for j, row in data.iterrows():
            vehicle_start, vehicle_end, class_ = row.values
            contains_vehicle = (audio_start < vehicle_start and vehicle_end < audio_end)
                
                
            if contains_vehicle:
                #assert vehicle_start - audio_start > 0, "invalid start time"
                #assert vehicle_end - audio_start < 10, "invalid end time"
                df.loc[len(df)] = {"onset": max(0, vehicle_start - audio_start - buffer), "offset": min(10, vehicle_end - audio_start + buffer), "class": class_}
                active_zone = True
        
        if active_zone:
            onset = int(floor((audio_start)*SR))
            offset = int(floor((audio_end)*SR))
            if audio_end < len_audio_sec * 0.8:
                train_path = './classifier/data/data/kauto5cls/train/'
                count = str(train_num).zfill(6)
                sf.write(f"{train_path}{name_audio}_train_{count}.wav", y[onset:offset], SR)  # slice by seconds, seconds in audio = SR * seconds
                df.to_csv(f"{train_path}{name_audio}_train_{count}.csv")
                train_num += 1
            elif audio_end > len_audio_sec * 0.8 and audio_end < len_audio_sec * 0.9:
                val_path = './classifier/data/data/kauto5cls/val/'
                count = str(val_num).zfill(6)
                sf.write(f"{val_path}{name_audio}_val_{count}.wav", y[onset:offset], SR)  # slice by seconds, seconds in audio = SR * seconds
                df.to_csv(f"{val_path}{name_audio}_val_{count}.csv")
                val_num += 1
            else:
                test_path = './classifier/data/data/kauto5cls/test/'
                count = str(test_num).zfill(6)
                sf.write(f"{test_path}{name_audio}_test_{count}.wav", y[onset:offset], SR)  # slice by seconds, seconds in audio = SR * seconds
                df.to_csv(f"{test_path}{name_audio}_test_{count}.csv")
                test_num += 1
        audio_start = audio_end
        audio_end += 10


if __name__ == "__main__":
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("-a", "--audio", default=None, type=str, help="Audio filename")
    parser.add_argument("-d", "--data", default=None, type=str, help="Audio .csv file")
    parser.add_argument("-c", "--class", default="class_1", type=str, help="Class in dataframe")
    args = vars(parser.parse_args())
    audio = args["audio"]
    data = args["data"]
    class_ = args["class"]
    assert audio is not None and data is not None, "No data or audio provided"
    name_audio = audio.split("/")[-1].split(".")[0]
    if name_audio == "": name_audio = audio
    path = str(Path.cwd())
    audio_path = path + "/" + audio
    data_path = path + "/" + data
    if not os.path.exists("./classifier/data/data/kauto5cls/train"): 
        os.makedirs("./classifier/data/data/kauto5cls/train")
    if not os.path.exists("./classifier/data/data/kauto5cls/val"): 
        os.makedirs("./classifier/data/data/kauto5cls/val")
    if not os.path.exists("./classifier/data/data/kauto5cls/test"): 
        os.makedirs("./classifier/data/data/kauto5cls/test")
    split_to_dirs(data_path, audio_path, class_, name_audio)
