import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Simulation settings
time_duration = 2 * 60  # 20 minutes in seconds
time_step = 0.02          # Simulation time step in seconds
time = np.arange(0, time_duration, time_step)

# Period settings (longitudinal: 5 sec, lateral: 7 sec)
longitudinal_period = 59
lateral_period = 71

# Longitudinal and lateral inputs (sine function)
accel =  7 * np.sin(2 * np.pi * time / longitudinal_period)
steer = 0.41 * np.sin(2 * np.pi * time / lateral_period)

# Vehicle parameters
mass = 3.5        # Vehicle mass (kg)
Lf = 0.15         # Distance from CG to front axle (m)
Lr = 0.175        # Distance from CG to rear axle (m)

# Pacejka tire model parameters
Bf, Cf, Df = 1.5, 1.5, 35
Br, Cr, Dr = 1.5, 1.5, 35
Cm0 = 0.04
# Yaw moment of inertia (kg*m^2)
Iz = 0.1          

# Initialize state variables [px, py, yaw, v, beta, omega]
state = np.zeros((len(time), 6))  # Columns: [px, py, yaw, v, beta, omega]
state[0, 3] = 2  # Initial velocity (m/s)

# Simulation loop
for i in range(1, len(time)):
    dt = time_step

    # Extract current states
    px, py, yaw, v, beta, omega = state[i-1]

    v=v+1e-5

    # Slip angles for front and rear tires
    alphaf = steer[i] - np.arctan2(Lf * omega + v * np.sin(beta), v * np.cos(beta))
    alphar = - np.arctan2( v * np.sin(beta)-Lr*omega, v * np.cos(beta))

    # Tire lateral forces using Pacejka model
    Ffy = Df * np.sin(Cf * np.arctan(Bf * alphaf))
    Fry = Dr * np.sin(Cr * np.arctan(Br * alphar))

    # Vehicle dynamics
    x_dot = v * np.cos(yaw + beta)
    y_dot = v * np.sin(yaw + beta)
    yaw_dot = omega
    v_dot = accel[i] * (1 - Cm0 * v)
    beta_dot = (Ffy + Fry) / (mass * v) - omega
    omega_dot = (1 / Iz) * (Ffy * Lf * np.cos(steer[i]) - Fry * Lr)

    # Update states using Euler's method
    state[i, 0] = px + x_dot * dt           # px
    state[i, 1] = py + y_dot * dt           # py
    state[i, 2] = yaw + yaw_dot * dt        # yaw
    state[i, 3] = v + v_dot * dt            # v
    state[i, 4] = beta + beta_dot * dt      # beta
    state[i, 5] = omega + omega_dot * dt    # omega

    state[i, 2] = (state[i, 2] + np.pi) % (2 * np.pi) - np.pi  # Normalize yaw angle

# Save results to DataFrame
df = pd.DataFrame({
    'time': time,
    'px': state[:, 0],
    'py': state[:, 1],
    'yaw': state[:, 2],
    'v': state[:, 3],
    'slip_angle': state[:, 4],
    'omega': state[:, 5],
    'accel': accel,
    'steer': steer
})

# Save to CSV
df.to_csv('data_gen.csv', index=False)

# Visualization
plt.figure(figsize=(12, 8))

# Plot position (px, py)
plt.subplot(2, 3, 1)
plt.plot(df['px'], df['py'], label="Trajectory")
plt.xlabel('Position X (m)')
plt.ylabel('Position Y (m)')
plt.title('Vehicle Trajectory')
plt.legend()

# Plot velocity
plt.subplot(2, 3, 2)
plt.plot(df['time'], df['v'], label="Velocity")
plt.xlabel('Time (s)')
plt.ylabel('Velocity (m/s)')
plt.title('Vehicle Velocity Over Time')
plt.legend()

# Plot yaw angle
plt.subplot(2, 3, 3)
plt.plot(df['time'], df['yaw'], label="Yaw Angle")
plt.xlabel('Time (s)')
plt.ylabel('Yaw (rad)')
plt.title('Yaw Angle Over Time')
plt.legend()

# Plot beta (sideslip angle)
plt.subplot(2, 3, 4)
plt.plot(df['time'], df['slip_angle'], label="Sideslip Angle (Beta)")
plt.xlabel('Time (s)')
plt.ylabel('Beta (rad)')
plt.title('Sideslip Angle Over Time')
plt.legend()


plt.subplot(2, 3, 5)
plt.plot(df['time'], df['steer'], label="Steering Angle")
plt.xlabel('Time (s)')
plt.ylabel('Steering Angle (rad)')
plt.title('Steering Angle Over Time')
plt.legend()


plt.subplot(2, 3, 6)
plt.plot(df['time'], df['accel'], label="Acceleration")
plt.xlabel('Time (s)')
plt.ylabel('Acceleration (m/s^2)')
plt.title('Acceleration Over Time')
plt.legend()


plt.tight_layout()
plt.show()