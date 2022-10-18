import pandas as pd

from tdt4290.classify_lidar import classify_lidar
from tdt4290.parse_lidar import parse_lidar_data

from .label_audio_with_lidar import label_audio_with_lidar


def main():
    audio_start = pd.Timestamp("2022-09-22 10:00:02.164")
    lidar = parse_lidar_data("data/lidar_log_grilstad_22.09.22_10_00_13_00.txt")
    classified_lidar = classify_lidar(lidar)
    labeled = label_audio_with_lidar(
        classified_lidar,
        audio_start,
    )
    print(labeled)
    labeled.to_csv("lidar_data_with_audio_timestamps.csv")


if __name__ == "__main__":
    main()
