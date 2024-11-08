import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

# Constants
DT = 0.02
CM0 = 0.04
MASS = 3.5
LF = 0.175
LR = 0.15

# Load CSV data
df = pd.read_csv("deep_dynamics/csv/default_params.csv") 

# Define the function to compute state update
def compute_state_update(v, slip_angle, omega, steer, accel, params):
    """Compute the updated state of the vehicle."""
    Iz, Df, Cf, Bf, Dr, Cr, Br = params
    
    # Calculate state derivatives
    v_dot = accel * (1 - v * CM0)
    alphaf = steer - np.arctan2(LF * omega + v * np.sin(slip_angle), v * np.cos(slip_angle))
    alphar = -np.arctan2(v * np.sin(slip_angle) - LR * omega, v * np.cos(slip_angle))
    
    Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
    Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))
    
    slip_angle_dot = (Ffy + Fry) / (MASS * v) - omega
    omega_dot = (1 / Iz) * (Ffy * LF * np.cos(steer) - Fry * LR)
    
    # Update states
    v_new = v + v_dot * DT
    slip_angle_new = slip_angle + slip_angle_dot * DT
    omega_new = omega + omega_dot * DT
    
    return np.array([v_new, slip_angle_new, omega_new])

# Define the RMSE calculation function
def calculate_rmse(params, data):
    """Calculate RMSE for the given parameters over all data."""
    squared_errors = []

    for _, row in data.iterrows():
        v, slip_angle, omega = row['v'], row['slip_angle'], row['omega']
        steer, accel = row['steer'], row['accel']
        
        # Predicted state
        predicted_state = compute_state_update(v, slip_angle, omega, steer, accel, params)
        actual_state = np.array([v, slip_angle, omega])
        
        squared_errors.append((actual_state - predicted_state) ** 2)
    
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

# Define parameter bounds based on reasonable ranges
bounds = [
    (0.05, 0.5),  # Iz bounds
    (20, 60),     # Df bounds
    (0.5, 5.0),   # Cf bounds
    (0.5, 5.0),   # Bf bounds
    (20, 60),     # Dr bounds
    (0.5, 5.0),   # Cr bounds
    (0.5, 5.0)    # Br bounds
]

# Create a callback to track progress and current RMSE
iteration_count = 0
MAX_ITERATIONS = 100

def progress_callback(x, convergence):
    """Callback to track the progress and RMSE during optimization."""
    global iteration_count
    iteration_count += 1
    progress_percent = (iteration_count / MAX_ITERATIONS) * 100
    current_rmse = calculate_rmse(x, df)
    print(f"Progress: {progress_percent:.2f}%, Current RMSE: {current_rmse:.6f}")

# Run differential evolution to minimize the RMSE
result = differential_evolution(
    func=calculate_rmse, 
    bounds=bounds, 
    args=(df,), 
    strategy='best1bin', 
    maxiter=MAX_ITERATIONS, 
    tol=1e-6,
    callback=progress_callback
)

# Output the best parameters and RMSE
optimal_params = result.x
print("Optimal parameters:")
param_names = ["Iz", "Df", "Cf", "Bf", "Dr", "Cr", "Br"]
for name, value in zip(param_names, optimal_params):
    print(f"{name}: {value}")
print("Minimum RMSE:", result.fun)
