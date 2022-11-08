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
):
    parsed_lidar = parse_lidar(filename, input_column_count)
    timestamped_lidar = timestamp_lidar(
        parsed_lidar, audio_start, vehicle_clip_length_s, audio_length
    )
    classified_lidar = classify_lidar(timestamped_lidar, classes)
    return classified_lidar
