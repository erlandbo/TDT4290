import pandas as pd

CLASSES = {
    "small": {
        "small": 100
    },
    "medium": {
        "medium_small": 160,
        "medium": 170,
        "medium_big":210,
    },
    "large": {
        "large": 9999
    }
}


def classify_lidar(lidar_data: pd.DataFrame):
    lidar_data["class_1"] = None
    lidar_data["class_2"] = None
    for class_1_key, subClasses in CLASSES.items(): 
        for class_2_key, value in subClasses.items(): 
            lidar_data.loc[(lidar_data["width"] <= value) & (lidar_data["class_1"].isnull()), "class_1"] = class_1_key
            lidar_data.loc[(lidar_data["width"] <= value) & (lidar_data["class_2"].isnull()), "class_2"] = class_2_key
    return lidar_data

