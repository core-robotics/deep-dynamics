#!/usr/bin/env python3

import numpy as np
import sys
from tqdm import tqdm
import csv

SAMPLING_TIME = 0.025


def write_dataset(csv_path, horizon, save=True):
    with open(csv_path) as f:
        csv_reader = csv.reader(f, delimiter=",")
        odometry = []
        throttle_cmds = []
        steering_cmds = []
        poses = []
        column_idxs = dict()
        previous_throttle = 0.0
        previous_steer = 0.0
        started = False
        for row in csv_reader:
            if len(column_idxs) == 0:
                for i in range(len(row)):
                    column_idxs[row[i].split("(")[0]] = i
                continue

            # vx = float(row[column_idxs["vx"]])
            v = float(row[column_idxs["v"]])

            if abs(v) < 0.3:
                if started:
                    break
                previous_throttle = float(row[column_idxs["accel"]])
                previous_steer = float(row[column_idxs["steer"]])
                continue
            # vy = float(row[column_idxs["vy"]])
            slip_angle = float(row[column_idxs["slip_angle"]])
            omega = float(row[column_idxs["omega"]])
            steering = float(row[column_idxs["steer"]])
            throttle = float(row[column_idxs["accel"]])

            steering_cmd = steering - previous_steer
            throttle_cmd = throttle - previous_throttle
            odometry.append(np.array([v, slip_angle, omega, throttle, steering]))
            poses.append
            (
                [
                    float(row[column_idxs["px"]]),
                    float(row[column_idxs["py"]]),
                    float(row[column_idxs["yaw"]]),
                    v,
                    slip_angle,
                    omega,
                    throttle,
                    steering,
                ]
            )
            previous_throttle += throttle_cmd
            previous_steer += steering_cmd
            if started:
                throttle_cmds.append(throttle_cmd)
                steering_cmds.append(steering_cmd)
            started = True
        odometry = np.array(odometry)
        throttle_cmds = np.array(throttle_cmds)
        steering_cmds = np.array(steering_cmds)
        features = np.zeros(
            (len(throttle_cmds) - horizon - 1, horizon, 8), dtype=np.double
        )
        labels = np.zeros((len(throttle_cmds) - horizon - 1, 3), dtype=np.double)

        for i in tqdm(
            range(len(throttle_cmds) - horizon - 1 - 5), desc="Compiling dataset"
        ):
            features[i] = np.array(
                [
                    *odometry[i : i + horizon].T,
                    throttle_cmds[i : i + horizon],
                    steering_cmds[i : i + horizon],
                    odometry[i + 5 : i + horizon + 5, 0],
                ]
            ).T
            labels[i] = np.array([*odometry[i + horizon]])[:3]
        poses = np.array(poses)
        print("Final features shape:", features.shape)
        print("Final labels shape:", labels.shape)
        if save:
            np.savez(
                csv_path[: csv_path.find(".csv")] + "_" + str(horizon) + ".npz",
                features=features,
                labels=labels,
                poses=poses,
            )
        return features, labels, poses


if __name__ == "__main__":
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description="Convert CSV file to pickled dataset")
    parser.add_argument("csv_path", type=str, help="CSV file to convert")
    parser.add_argument("horizon", type=int, help="Horizon of timestamps used")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict: dict = vars(args)
    write_dataset(argdict["csv_path"], argdict["horizon"])
