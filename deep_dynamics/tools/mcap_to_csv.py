import csv
import struct
from mcap.reader import make_reader
from tqdm import tqdm  

# file path
bagfile_path = "/home/a/Downloads/rosbag2_2024_10_14-14_47_22/rosbag2_2024_10_14-14_47_22_0.mcap"
output_csv_path = "/home/a/deep-dynamics/deep_dynamics/csv/rosbag2_2024_10_14.csv"

# message format and fieldnames
MESSAGE_FORMAT = "15d"  # float64 , 15 data fields
fieldnames = [
    'timestamp', 'px', 'py', 'yaw', 'v', 'vx', 'vy', 'omega',
    'a', 'ax', 'ay', 'slip_angle', 'accel', 'jerk', 'steer', 'steer_vel'
]

# MCAP file read and write to CSV
with open(bagfile_path, 'rb') as f, open(output_csv_path, 'w', newline='') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()

    message_reader = make_reader(f)
    total_messages = sum(1 for _ in message_reader.iter_messages())  # 메시지 개수 계산
    
    
    f.seek(0) 
    message_reader = make_reader(f)

    for schema, channel, message in tqdm(message_reader.iter_messages(), total=total_messages, desc="Processing MCAP messages"):
        if "CarState" in schema.name:
            # 4byte header excluded
            data = struct.unpack(MESSAGE_FORMAT, message.data[4:])
            # Write the extracted data to CSV
            writer.writerow(dict(zip(fieldnames, [message.log_time] + list(data))))

print(f"Data successfully written to {output_csv_path}")
