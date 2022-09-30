from io import StringIO
import pandas as pd


def _load_lidar_data_as_string(filename: str):
    with open(filename) as f:
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

    lidar_data = raw_lidar_data.filter([3, 4, 8, 9, 35, 39, 40])
    lidar_data.rename(
        columns=(
            {
                3: "enter_date",
                4: "enter_time",
                8: "leave_date",
                9: "leave_time",
                35: "y0",
                39: "y1",
                40: "height",
            }
        ),
        inplace=True,
    )
    lidar_data["width"] = lidar_data["y1"] - lidar_data["y0"]
    # return lidar_data[lidar_data[5] == "mldcs.VehicleDetector[0]"].to_dict()
    return lidar_data[::2]  # .to_dict("records")


def parse_lidar_data(filename: str) -> pd.DataFrame:  # is a List[LidarData]
    text_lidar_data = _load_lidar_data_as_string(filename)
    return _parse_lidar_data(text_lidar_data)
