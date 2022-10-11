import pandas as pd

CLASSES = {
    "small": {"small": 100},
    "medium": {
        "medium_small": 160,
        "medium": 170,
        "medium_big": 210,
    },
    "large": {"large": 9999},
}


def classify_lidar(lidar_data: pd.DataFrame):
    lidar_data_copy = lidar_data.copy()
    lidar_data_copy["class_1"] = None
    lidar_data_copy["class_2"] = None
    for class_1_key, sub_classes in CLASSES.items():
        for class_2_key, value in sub_classes.items():
            lidar_data_copy.loc[
                (lidar_data_copy["width"] <= value)
                & (lidar_data_copy["class_1"].isnull()),
                "class_1",
            ] = class_1_key
            lidar_data_copy.loc[
                (lidar_data_copy["width"] <= value)
                & (lidar_data_copy["class_2"].isnull()),
                "class_2",
            ] = class_2_key
    return lidar_data_copy
