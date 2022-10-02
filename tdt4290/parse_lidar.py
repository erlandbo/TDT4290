from io import StringIO

import pandas as pd


def _load_lidar_data_as_string(filename: str):
    with open(filename, "r", encoding="ISO-8859-1") as f:
        text_lidar_data = f.read().strip('\\"')
    return text_lidar_data


def _parse_lidar_data(lidar_data_string: str) -> pd.DataFrame:
    raw_lidar_data = pd.read_csv(
        StringIO(lidar_data_string),
        sep=" ",
        engine="python",
        quoting=3,
        on_bad_lines="skip",  # TODO: Handle this
        header=None,
    )

    lidar_data = raw_lidar_data.filter([27, 28, 29, 30, 35, 39, 40])
    lidar_data.rename(
        columns=(
            {
                27: "enter_date",
                28: "enter_time",
                29: "leave_date",
                30: "leave_time",
                35: "y0",
                39: "y1",
                40: "height",
            }
        ),
        inplace=True,
    )
    # return lidar_data[lidar_data[5] == "mldcs.VehicleDetector[0]"].to_dict()
    return lidar_data[::2]  # .to_dict("records")


def _add_features_to_lidar(
    raw_lidar_data: pd.DataFrame, start_time: pd.Timestamp
) -> pd.DataFrame:
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
    lidar_data["datetime_enter"] = pd.to_datetime(
        lidar_data["enter_date"] + " " + lidar_data["enter_time"]
    )

    lidar_data["datetime_leave"] = pd.to_datetime(
        lidar_data["leave_date"] + " " + lidar_data["leave_time"]
    )
    lidar_data["seconds_enter"] = (
        pd.to_datetime(lidar_data["enter_date"] + " " + lidar_data["enter_time"])
        - start_time
    ).dt.total_seconds()
    lidar_data["seconds_leave"] = (
        pd.to_datetime(lidar_data["leave_date"] + " " + lidar_data["leave_time"])
        - start_time
    ).dt.total_seconds()
    lidar_data.drop(
        ["enter_date", "enter_time", "leave_date", "leave_time"],
        axis=1,
        inplace=True,
    )
    lidar_data["width"] = lidar_data["y1"] - lidar_data["y0"]
    lidar_data = lidar_data[::2]
    lidar_data["duration"] = lidar_data["datetime_leave"] - lidar_data["datetime_enter"]
    lidar_data["front_area"] = lidar_data["width"] * lidar_data["height"]
    return lidar_data


def parse_lidar_data(
    filename: str, start_time: pd.Timestamp
) -> pd.DataFrame:  # is a List[LidarData]
    text_lidar_data = _load_lidar_data_as_string(filename)
    raw_lidar_data = _parse_lidar_data(text_lidar_data)
    return _add_features_to_lidar(raw_lidar_data, start_time)
