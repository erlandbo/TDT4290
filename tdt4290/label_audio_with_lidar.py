from typing import Optional

import pandas as pd


def label_audio_with_lidar(
    lidar: pd.DataFrame,
    audio_start: pd.Timestamp,  # = pd.to_datetime("2022-09-22 10:05:10.0000"),
    vehicle_clip_length_s: int = 2,
    audio_length: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    lidar["timedelta_start"] = lidar["datetime_enter"] - audio_start
    lidar["timedelta_end"] = lidar["datetime_leave"] - audio_start
    absolute_start = lidar["timedelta_start"].dt.total_seconds()
    absolute_end = lidar["timedelta_end"].dt.total_seconds()
    diff_half = (absolute_end - absolute_start) / 2
    lidar["audio_start_s"] = (
        absolute_start + diff_half - (vehicle_clip_length_s / 2)
    ).astype("int")
    lidar["audio_end_s"] = (
        absolute_end - diff_half + (vehicle_clip_length_s / 2)
    ).astype("int")
    if audio_length is None:
        return lidar
    audio_end = audio_start + audio_length
    return lidar.loc[
        (lidar["datetime_enter"] >= audio_start)
        & (lidar["datetime_leave"] <= audio_end)
    ]
