from typing import TypedDict


class LidarData(TypedDict):
    enter_date: str
    enter_time: str
    leave_date: str
    leave_time: str
    y0: int  # in cm
    y1: int  # in cm
    height: int  # in cm
    width: int  # in cm
