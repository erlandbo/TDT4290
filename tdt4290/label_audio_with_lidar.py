from typing import List

import librosa
import pandas as pd

from .parse_lidar import parse_lidar_data
from .types_common import LidarLabeledAudio


def label_audio_with_lidar(
    audio_filename: str,
    lidar_filename: str,
    audio_start: pd.Timestamp = pd.to_datetime("2022-09-22 10:00:02.334612"),
) -> List[LidarLabeledAudio]:
    print("Parsing lidar data...")
    lidar = parse_lidar_data(lidar_filename, audio_start)
    print("Done parsing lidar data")
    print("Start loading audio...")
    sound, sample_rate = librosa.load(audio_filename)
    lidar_labeled_audio: List[LidarLabeledAudio] = []
    for i, row in lidar.iterrows():
        enter = int(row["seconds_enter"] * sample_rate)
        leave = int(row["seconds_leave"] * sample_rate)
        item: LidarLabeledAudio = {
            "lidar_data": row.to_dict().__str__(),
            "audio_sample": sound[leave:enter].tolist(),  # sound[enter:leave],
        }
        lidar_labeled_audio.append(item)
    return lidar_labeled_audio
