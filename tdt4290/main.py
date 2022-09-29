from parse_lidar import parse_lidar


def main():
    data = parse_lidar("data/lidar_log_grilstad_22.09.22_10_00_13_00.txt")
    print("Hello")
    print(data)


if __name__ == "__main__":
    main()
