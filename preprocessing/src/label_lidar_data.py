from typing import Dict, Optional

import pandas as pd

from .classify_lidar_data import CLASSES, classify_lidar
from .parse_lidar_data import parse_lidar
from .timestamp_lidar_data import timestamp_lidar


def label_lidar(
    filename: str,
    audio_start: pd.Timestamp,
    vehicle_clip_length_s=2,
    input_column_count: int = 48,
    audio_length: Optional[pd.Timedelta] = None,
    classes: Dict[str, Dict[str, int]] = CLASSES,
) -> pd.DataFrame:
    """Parses, timestamps and classifies the vehicles in a lidar data .txt file.

    Args:
        filename (str): The path to the .txt file of lidar data.
        audio_start (pd.Timestamp): The start time for the audio to synchronize
            the audio time and the lidar data time.
        vehicle_clip_length_s (int, optional): How long the timestamp for each vehicle
            should be in seconds. Defaults to 2.
        input_column_count (int, optional): The count of columns in the lidar .txt file.
            Defaults to 48.
        audio_length (Optional[pd.Timedelta], optional): The length of the whole audio
            file to limit the size of the output data frame. Defaults to None.
        classes (Dict[str, Dict[str, int]], optional): The classes to classify
            the vehicles with. Defaults to
            {
                "small": {"small": 100},
                "medium": {
                    "medium_small": 160,
                    "medium": 170,
                    "medium_big": 210,
                },
                "large": {"large": 9999},
            }

    Returns:
        pd.DataFrame: A pandas DataFrame with columns
            datetime_enter, datetime_leave, y0, y1, height, width, front_area, duration,
            audio_start_s, audio_end_s, class_1, class_2
    """
    parsed_lidar = parse_lidar(filename, input_column_count)
    timestamped_lidar = timestamp_lidar(
        parsed_lidar, audio_start, vehicle_clip_length_s, audio_length
    )
    classified_lidar = classify_lidar(timestamped_lidar, classes)
    return classified_lidar
