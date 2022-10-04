from typing import Optional

import pandas as pd

from .parse_lidar import parse_lidar_data


def label_audio_with_lidar(
    lidar_filename: str,
    audio_start: pd.Timestamp,  # = pd.to_datetime("2022-09-22 10:05:10.0000"),
    audio_sample_rate: int = 32000,
    vehicle_clip_length_s: int = 2,
    audio_length: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    lidar: pd.DataFrame = parse_lidar_data(lidar_filename)
    lidar["timedelta_start"] = lidar["datetime_enter"] - audio_start
    lidar["timedelta_end"] = lidar["datetime_leave"] - audio_start
    absolute_start = lidar["timedelta_start"].dt.total_seconds() * audio_sample_rate
    absolute_end = lidar["timedelta_end"].dt.total_seconds() * audio_sample_rate
    diff_half = (absolute_end - absolute_start) / 2
    lidar["audio_start_index"] = (
        absolute_start + diff_half - (vehicle_clip_length_s / 2) * audio_sample_rate
    ).astype("int")
    lidar["audio_end_index"] = (
        absolute_end - diff_half + (vehicle_clip_length_s / 2) * audio_sample_rate
    ).astype("int")
    if audio_length is None:
        return lidar
    audio_end = audio_start + audio_length
    return lidar.loc[
        (lidar["datetime_enter"] >= audio_start)
        & (lidar["datetime_leave"] <= audio_end)
    ]
