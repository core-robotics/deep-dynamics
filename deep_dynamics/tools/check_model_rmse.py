import numpy as np
import pandas as pd

# Constants
dt = 0.02
Cm0 = 0.04
mass = 3.5
lf = 0.175
lr = 0.15

# Default parameters
# Iz = 0.2
# Df, Cf, Bf = 36, 1.5, 1.2
# Dr, Cr, Br = 45, 1.5, 1.2

# C-> 1.5 constraints
Iz = 0.22
# Iz = 0.17
Df, Cf, Bf = 30.20, 1.5, 1.09
Dr, Cr, Br = 51.07, 1.5, 1.53

# full boundaries
# Iz = 0.24
# Df, Cf, Bf = 31.11, 1.19, 1.26
# Dr, Cr, Br = 46.87, 1.71, 1.40


# Load CSV data
df = pd.read_csv("deep_dynamics/csv/default_params.csv")  # Replace with your actual file path

# Initialize lists to store the hypothetical values and errors
X_hat_values = []
squared_errors = []

# Function to calculate X_hat
def calculate_X_hat(v, slip_angle, omega, steer, accel):
    v_dot = accel * (1 - v * Cm0)
    beta = slip_angle
    alphaf = steer - np.arctan2(lf * omega + v * np.sin(beta), v * np.cos(beta))
    alphar = -np.arctan2(v * np.sin(beta) - lr * omega, v * np.cos(beta))
    Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
    Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))
    slip_angle_dot = (Ffy + Fry) / (mass * v) - omega
    omega_dot = (1 / Iz) * (Ffy * lf * np.cos(steer) - Fry * lr)
    v_new = v + v_dot * dt
    slip_angle_new = slip_angle + slip_angle_dot * dt
    omega_new = omega + omega_dot * dt
    return np.array([v_new, slip_angle_new, omega_new])

# Loop through each row to calculate hypothetical X_hat and RMSE
for i, row in df.iterrows():
    v, slip_angle, omega = row['v'], row['slip_angle'], row['omega']
    steer, accel = row['steer'], row['accel']
    X = np.array([v, slip_angle, omega])
    X_hat = calculate_X_hat(v, slip_angle, omega, steer, accel)
    X_hat_values.append(X_hat)
    squared_errors.append((X - X_hat) ** 2)

# Convert list of squared errors to an array and calculate RMSE
squared_errors = np.array(squared_errors)
rmse = np.sqrt(squared_errors.mean(axis=0))

print("RMSE for [v, slip_angle, omega]:", rmse)
print("Average RMSE:", rmse.mean())
