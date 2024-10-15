import csv
import struct
from mcap.reader import make_reader

# Path to your MCAP file
bagfile_path = "/home/a/deep-dynamics/deep_dynamics/bag/rosbag2_2024_10_14-14_47_22/rosbag2_2024_10_14-14_47_22_0.mcap"
output_csv_path = "/home/a/deep-dynamics/deep_dynamics/csv/rosbag2_2024_10_14.csv"

# Define the message structure (based on your CarState message definition)
# Each field is a 64-bit (8-byte) float (float64), and there are 14 fields
MESSAGE_FORMAT = "15d"  # 'd' means double (float64), repeated 14 times

# Field names from the CarState message
fieldnames = [
    'timestamp', 'px', 'py', 'yaw', 'v', 'vx', 'vy', 'omega',
    'a', 'ax', 'ay', 'slip_angle', 'accel', 'jerk', 'steer', 'steer_vel'
]

# Open the MCAP file and read the data
with open(bagfile_path, 'rb') as f:
    # Initialize MCAP reader
    mcap_reader = make_reader(f)
    
    # Prepare CSV file to write
    with open(output_csv_path, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header
        writer.writeheader()

        # Iterate over messages in the MCAP file
        for schema, channel, message in mcap_reader.iter_messages():
            # Check if the message type matches our custom message type (manually check schema)
            if "CarState" in schema.name:
                message_data = message.data[4:] 
                # Unpack the message data according to the defined structure (14 float64 fields)
                # print(f"Message size: {len(message.data)} bytes")
                # print(f"Message data: {len(message_data)} bytes")
                # data = struct.unpack(MESSAGE_FORMAT, message.data)
                data=struct.unpack(MESSAGE_FORMAT, message_data)
                # print(f"Data: {data}")
                # Write the extracted data to CSV
                writer.writerow({
                    'timestamp': message.log_time,
                    'px': data[0],
                    'py': data[1],
                    'yaw': data[2],
                    'v': data[3],
                    'vx': data[4],
                    'vy': data[5],
                    'omega': data[6],
                    'a': data[7],
                    'ax': data[8],
                    'ay': data[9],
                    'slip_angle': data[10],
                    'accel': data[11],
                    'jerk': data[12],
                    'steer': data[13],
                    'steer_vel': data[14],
                })

print(f"Data successfully written to {output_csv_path}")
