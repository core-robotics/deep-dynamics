import numpy as np
import matplotlib.pyplot as plt

def pacejka_lateral_force(alpha, B, C, D):
    """Calculate lateral force using Pacejka tire model."""
    return D * np.sin(C * np.arctan(B * alpha))

def generate_tire_data(alpha_range, params):
    """Generate lateral force data for given slip angle range and tire parameters."""
    B, C, D = params
    return pacejka_lateral_force(alpha_range, B, C, D)

def plot_lateral_forces(alphaf, alphar, Ffy, Ffy_alt, Fry, Fry_alt, front_params, front_params_alt, rear_params, rear_params_alt):
    """Plot lateral forces for front and rear tires in a structured layout with parameter annotations."""
    plt.figure(figsize=(14, 6))

    # Extract parameter values for annotation
    Bf, Cf, Df = front_params
    Bf_alt, Cf_alt, Df_alt = front_params_alt
    Br, Cr, Dr = rear_params
    Br_alt, Cr_alt, Dr_alt = rear_params_alt

    # Front tire lateral force comparison with parameters in title
    plt.subplot(1, 2, 1)
    plt.plot(alphaf * 180 / np.pi, Ffy, label=f'Front Tire (Original) B={Bf}, C={Cf}, D={Df}')
    plt.plot(alphaf * 180 / np.pi, Ffy_alt, label=f'Front Tire (Modified) B={Bf_alt}, C={Cf_alt}, D={Df_alt}')
    plt.xlabel('Slip Angle (degrees)')
    plt.ylabel('Lateral Force (N)')
    plt.title(f'Front Tire Lateral Force')
    plt.legend()
    plt.grid(True)

    # Rear tire lateral force comparison with parameters in title
    plt.subplot(1, 2, 2)
    plt.plot(alphar * 180 / np.pi, Fry, label=f'Rear Tire (Original) B={Br}, C={Cr}, D={Dr}')
    plt.plot(alphar * 180 / np.pi, Fry_alt, label=f'Rear Tire (Modified) B={Br_alt}, C={Cr_alt}, D={Dr_alt}')
    plt.xlabel('Slip Angle (degrees)')
    plt.ylabel('Lateral Force (N)')
    plt.title(f'Rear Tire Lateral Force')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

# Tire parameters (B, C, D) for original and modified settings
front_tire_params = (1.5, 1.5, 35)
rear_tire_params = (1.5, 1.5, 35)
front_tire_params_alt = (1.2, 1.5, 35)
rear_tire_params_alt = (1.2, 1.5, 35)

# Slip angle range in radians (-50 to 50 degrees)
alphaf = np.linspace(-100, 100, 100) * np.pi / 180
alphar = np.linspace(-100, 100, 100) * np.pi / 180

# Generate lateral force data
Ffy = generate_tire_data(alphaf, front_tire_params)
Ffy_alt = generate_tire_data(alphaf, front_tire_params_alt)
Fry = generate_tire_data(alphar, rear_tire_params)
Fry_alt = generate_tire_data(alphar, rear_tire_params_alt)

# Plot the results with parameter annotations
plot_lateral_forces(alphaf, alphar, Ffy, Ffy_alt, Fry, Fry_alt, front_tire_params, front_tire_params_alt, rear_tire_params, rear_tire_params_alt)
