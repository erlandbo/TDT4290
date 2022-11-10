from typing import Optional

import pandas as pd


def timestamp_lidar(
    lidar_data: pd.DataFrame,
    audio_start: pd.Timestamp,
    vehicle_clip_length_s: int = 2,
    audio_length: Optional[pd.Timedelta] = None,
) -> pd.DataFrame:
    """Add timestamps to lidar data based on the start time of the audio clip.

    Args:
        lidar (pd.DataFrame): The lidar data DataFrame
        audio_start (pd.Timestamp): The start time of the audio clip.
        vehicle_clip_length_s (int, optional): How long the timestamp for each vehicle
            should be in seconds. Defaults to 2.
        audio_length (Optional[pd.Timedelta], optional): Length of the audio clip
            to limit the size of the output DataFrame. If None will add timestamps
            based on the passing vehicles in the lidar data for all rows in the
            lidar data data frame. Defaults to None.

    Returns:
        pd.DataFrame: A copy of the input lidar_data with the added columns for
            timestamps: audio_start_s, audio_end_s
    """
    lidar = lidar_data.copy()
    lidar["timedelta_start"] = lidar["datetime_enter"] - audio_start
    lidar["timedelta_end"] = lidar["datetime_leave"] - audio_start
    absolute_start = lidar["timedelta_start"].dt.total_seconds()
    absolute_end = lidar["timedelta_end"].dt.total_seconds()
    diff_half = (absolute_end - absolute_start) / 2
    lidar["audio_start_s"] = (
        absolute_start + diff_half - (vehicle_clip_length_s / 2)
    ).astype("float")
    lidar["audio_end_s"] = (
        absolute_end - diff_half + (vehicle_clip_length_s / 2)
    ).astype("float")
    if audio_length is None:
        return lidar
    audio_end = audio_start + audio_length
    return lidar.loc[
        (lidar["datetime_enter"] >= audio_start)
        & (lidar["datetime_leave"] <= audio_end)
    ]
