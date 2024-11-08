import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Constants
dt = 0.02
Cm0 = 0.04
mass = 3.5
lf = 0.175
lr = 0.15

# Load CSV data
df = pd.read_csv("deep_dynamics/csv/default_params.csv")  # Replace with your actual file path

# Define the function to compute state update
def compute_state_update(v, slip_angle, omega, steer, accel, params):
    Iz, Df, Cf, Bf, Dr, Cr, Br = params
    
    v_dot = accel * (1 - v * Cm0)
    beta = slip_angle
    alphaf = steer - np.arctan2(lf * omega + v * np.sin(beta), v * np.cos(beta))
    alphar = -np.arctan2(v * np.sin(beta) - lr * omega, v * np.cos(beta))
    
    Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
    Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))
    
    slip_angle_dot = (Ffy + Fry) / (mass * v) - omega
    omega_dot = (1 / Iz) * (Ffy * lf * np.cos(steer) - Fry * lr)
    
    # Update the states
    v_new = v + v_dot * dt
    slip_angle_new = slip_angle + slip_angle_dot * dt
    omega_new = omega + omega_dot * dt
    
    return np.array([v_new, slip_angle_new, omega_new])

# Define the RMSE calculation function
def calculate_rmse(params, df):
    squared_errors = []
    
    for _, row in df.iterrows():
        v, slip_angle, omega = row['v'], row['slip_angle'], row['omega']
        steer, accel = row['steer'], row['accel']
        
        X_hat = compute_state_update(v, slip_angle, omega, steer, accel, params)
        X = np.array([v, slip_angle, omega])
        
        squared_errors.append((X - X_hat) ** 2)
    
    squared_errors = np.array(squared_errors)
    rmse = np.sqrt(squared_errors.mean(axis=0))
    return rmse.mean()

# Define parameter bounds based on reasonable ranges
bounds = [
    (0.05, 0.5),  # Iz bounds
    (20, 60),      # Df bounds
    (0.5, 5.0),    # Cf bounds
    (0.5, 5.0),    # Bf bounds
    (20, 60),      # Dr bounds
    (0.5, 5.0),    # Cr bounds
    (0.5, 5.0)     # Br bounds
]

# Create a callback to track progress and current RMSE
iteration_count = 0
max_iterations = 10

def progress_callback(x, convergence):
    global iteration_count
    iteration_count += 1
    progress_percent = (iteration_count / max_iterations) * 100
    current_rmse = calculate_rmse(x, df) 
    print(f"Progress: {progress_percent:.2f}%, Current RMSE: {current_rmse:.6f}")

# Run differential evolution to minimize the RMSE
result = differential_evolution(
    calculate_rmse, 
    bounds, 
    args=(df,), 
    strategy='best1bin', 
    maxiter=max_iterations, 
    tol=1e-6,
    callback=progress_callback
)

# Output the best parameters and RMSE
optimal_params = result.x
print("Optimal parameters:", optimal_params)
print("Minimum RMSE:", result.fun)
