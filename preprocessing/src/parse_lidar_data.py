from io import StringIO

import pandas as pd


def _load_lidar_data_as_string(filename: str):
    with open(filename, "r", encoding="ISO-8859-1") as f:
        text_lidar_data = (
            f.read().replace('\\"', "").replace('"', "").replace("  ", " ")
        )
    return text_lidar_data


def _parse_lidar_data(lidar_data_string: str, column_count: int) -> pd.DataFrame:
    raw_lidar_data = pd.read_csv(
        StringIO(lidar_data_string),
        sep=" ",
        engine="python",
        quoting=3,
        on_bad_lines="warn",
        names=[i for i in range(column_count + 1)],
        header=None,
    )

    lidar_data = raw_lidar_data[raw_lidar_data[5] == "mldcs.VehicleDetector[0]"]
    # lidar_data = raw_lidar_data
    # print("print", lidar_data)
    lidar_data = lidar_data.filter([27, 28, 29, 30, 34, 38, 39])
    lidar_data.rename(
        columns=(
            {
                27: "enter_date",
                28: "enter_time",
                29: "leave_date",
                30: "leave_time",
                34: "y0",
                38: "y1",
                39: "height",
            }
        ),
        inplace=True,
    )
    return lidar_data


def _add_features_to_lidar(raw_lidar_data: pd.DataFrame) -> pd.DataFrame:
    """Adds features to raw_lidar_data.

    Args:
        raw_lidar_data (pd.DataFrame): DataFrame containing columns
            enter_date, enter_time, leave_date, leave_time, y0, y1, height.

    Returns:
        pd.DataFrame: DataFrame containing columns:
            datetime_enter, datetime_leave, width, duration, front_area.
    """
    # date_format = "&y-&m-&d %H:%M:%S"
    lidar_data = raw_lidar_data.copy()
    lidar_data["width"] = lidar_data["y1"] - lidar_data["y0"]
    lidar_data["front_area"] = lidar_data["width"] * lidar_data["height"]
    lidar_data["datetime_enter"] = pd.to_datetime(
        lidar_data["enter_date"] + " " + lidar_data["enter_time"]
    )
    lidar_data["datetime_leave"] = pd.to_datetime(
        lidar_data["leave_date"] + " " + lidar_data["leave_time"]
    )
    lidar_data.drop(
        ["enter_date", "enter_time", "leave_date", "leave_time"],
        axis=1,
        inplace=True,
    )
    lidar_data["duration"] = lidar_data["datetime_leave"] - lidar_data["datetime_enter"]
    lidar_data["duration"] = lidar_data.duration.dt.total_seconds()
    return lidar_data


def parse_lidar(filename: str, column_count=48) -> pd.DataFrame:  # is a List[LidarData]
    """Parses a .txt file of lidar data to a Pandas data frame.

    Args:
        filename (str): The path to the lidar data .txt file.
        column_count (int, optional): The number of columns in the lidar data.
            Defaults to 48.

    Returns:
        pd.DataFrame: A pandas data frame to represent the lidar data with columns:
            datetime_enter, datetime_leave, y0, y1, height, width, front_area, duration.

    """
    text_lidar_data = _load_lidar_data_as_string(filename)
    raw_lidar_data = _parse_lidar_data(text_lidar_data, column_count)
    return _add_features_to_lidar(raw_lidar_data)
