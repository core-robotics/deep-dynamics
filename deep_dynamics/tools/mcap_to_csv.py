import csv
import struct
from mcap.reader import make_reader
from tqdm import tqdm
import numpy as np
from scipy.interpolate import interp1d

# File paths
bagfile_path = "/home/a/bag/rosbag2_2024_10_23-22_30_42/50hz_3.mcap"
output_csv_path = "/home/a/deep-dynamics/deep_dynamics/csv/sim_50hz_3.csv"

# Message format and fieldnames
MESSAGE_FORMAT = "16d"  # float64, 16 data fields
fieldnames = [
    "timestamp",
    "px",
    "py",
    "yaw",
    "v",
    "vx",
    "vy",
    "v_dot",
    "omega",
    "a",
    "ax",
    "ay",
    "slip_angle",
    "accel",
    "jerk",
    "steer",
    "steer_vel",
]

# Define the topic name to filter
target_topic = "/state0"
target_frequency = 50  # Frequency in Hz for interpolation

# MCAP file read and write to CSV
with open(bagfile_path, "rb") as f, open(output_csv_path, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    message_reader = make_reader(f)

    timestamps = []
    data_points = {field: [] for field in fieldnames[1:]}  # Skip 'timestamp' for data

    # Read all messages first
    for schema, channel, message in tqdm(
        message_reader.iter_messages(),
        desc="Reading MCAP messages",
    ):
        if "CarState" in schema.name and channel.topic == target_topic:
            # Extract data
            data = struct.unpack(MESSAGE_FORMAT, message.data[4:])
            timestamps.append(message.log_time * 1e-9)  # Convert timestamp to seconds
            for i, field in enumerate(fieldnames[1:]):  # Exclude 'timestamp' from fieldnames
                data_points[field].append(data[i])

    # Create regular timestamps at the desired frequency
    start_time = timestamps[0]
    end_time = timestamps[-1]
    total_duration = end_time - start_time
    num_interpolated_points = int(total_duration * target_frequency)

    regular_timestamps = np.linspace(start_time, end_time, num_interpolated_points)

    # Interpolation for each data field
    interpolated_data = {}
    for field in fieldnames[1:]:
        interpolator = interp1d(timestamps, data_points[field], kind='linear', fill_value="extrapolate")
        interpolated_data[field] = interpolator(regular_timestamps)

    # Write the interpolated data to CSV
    for i, ts in enumerate(regular_timestamps):
        row = {'timestamp': ts}
        for field in fieldnames[1:]:
            row[field] = interpolated_data[field][i]
        writer.writerow(row)

print(f"Interpolated data successfully written to {output_csv_path}")
