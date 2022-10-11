import json

import pandas as pd

from .label_audio_with_lidar import label_audio_with_lidar


def main():
    print("Start...")
    audio_start = pd.Timestamp("2022-09-22 10:00:02.164")
    labeled = label_audio_with_lidar(
        ".data/audio_22092022.WAV",
        "data/lidar_log_grilstad_22.09.22_10_00_13_00.txt",
        audio_start,
    )
    print("Done labeling")
    print(json.dumps(labeled))


if __name__ == "__main__":
    main()
