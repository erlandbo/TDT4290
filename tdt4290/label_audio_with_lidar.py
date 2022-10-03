from typing import List

import pandas as pd

from .parse_lidar import parse_lidar_data
from .types_common import LidarLabeledAudio


def label_audio_with_lidar(
    lidar_filename: str,
    audio_start: pd.Timestamp,  # = pd.to_datetime("2022-09-22 10:05:10.0000"),
    audio_sample_rate: int = 32000,
) -> List[LidarLabeledAudio]:
    lidar: pd.DataFrame = parse_lidar_data(lidar_filename)
    # audio_sample_start_list: List[int] = []
    # audio_sample_end_list: List[int] = []
    # for _, row in lidar.iterrows():
    #     enter: pd.Timedelta = row["datetime_enter"] - audio_start
    #     leave: pd.Timedelta = row["datetime_leave"] - audio_start
    #     if enter < first_car_enter_audio_time or (
    #         audio_end is not None and leave > audio_end
    #     ):
    #         audio_sample_start_list.append(-1)
    #         audio_sample_end_list.append(-1)
    #     audio_sample_start_list.append(enter.total_seconds() * audio_sample_rate)
    #     audio_sample_end_list.append(leave.total_seconds() * audio_sample_rate)
    lidar["audio_start_index"] = (
        lidar["datetime_enter"] - audio_start
    ).dt.total_seconds() * audio_sample_rate
    lidar["audio_end_index"] = (
        lidar["datetime_leave"] - audio_start
    ).dt.total_seconds() * audio_sample_rate

    return lidar
