# Import necessary packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load CSV data
df = pd.read_csv("deep_dynamics/csv/default_params.csv")

# Assuming columns `time`, `steer`, `accel`, `v`, `slip_angle`, and `omega` are present in the CSV
# time = df['timestamp']
steer_input = df['steer']
accel_input = df['accel']
velocity_state = df['v']
slip_angle_state = df['slip_angle']
omega_state = df['omega']
# DT = time[1] - time[0]  # Sampling period in seconds
DT = 0.02

# Define a function to calculate delay based on cross-correlation
def calculate_delay(control_signal, state_signal, DT):
    cross_corr = np.correlate(control_signal - control_signal.mean(), state_signal - state_signal.mean(), mode='full')
    delay_index = cross_corr.argmax() - (len(state_signal) - 1)
    delay_time = delay_index * DT
    return delay_time

# Calculate delays
steer_velocity_delay = calculate_delay(steer_input, velocity_state, DT)
steer_slip_angle_delay = calculate_delay(steer_input, slip_angle_state, DT)
steer_omega_delay = calculate_delay(steer_input, omega_state, DT)

accel_velocity_delay = calculate_delay(accel_input, velocity_state, DT)
accel_slip_angle_delay = calculate_delay(accel_input, slip_angle_state, DT)
accel_omega_delay = calculate_delay(accel_input, omega_state, DT)

# Display the calculated delays
print(f"Steer to Velocity Delay: {steer_velocity_delay:.2f} seconds")
print(f"Steer to Slip Angle Delay: {steer_slip_angle_delay:.2f} seconds")
print(f"Steer to Omega Delay: {steer_omega_delay:.2f} seconds")

print(f"Accel to Velocity Delay: {accel_velocity_delay:.2f} seconds")
print(f"Accel to Slip Angle Delay: {accel_slip_angle_delay:.2f} seconds")
print(f"Accel to Omega Delay: {accel_omega_delay:.2f} seconds")
