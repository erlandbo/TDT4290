from typing import Dict

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


def classify_lidar(
    lidar_data: pd.DataFrame, classes: Dict[str, Dict[str, int]] = CLASSES
) -> pd.DataFrame:
    """Adds the class_1 and class_2 columns on a lidar dataset of vehicles.
    Requires that the input data frame has the column "width" to classify from.

    Args:
        lidar_data (pd.DataFrame): The lidar data frame containing the width column.
        classes (Dict[str, Dict[str, int]], optional): A dictionary containing
            the name of the class_1 classes. Each class_1 class has a dictionary of
            the class_2 classes as values, that is, the subclasses of class_1's keys.
            The value of the subclasses are the maximum width of a vehicle to be
            considered that class. Vehicles are classified to the smallest class they
            fit within. Defaults to
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
        pd.DataFrame: A new data frame with the class_1 and class_2 classes added.
    """
    lidar_data_copy = lidar_data.copy()
    lidar_data_copy["class_1"] = None
    lidar_data_copy["class_2"] = None
    for class_1_key, sub_classes in classes.items():
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
