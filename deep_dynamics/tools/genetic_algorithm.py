import numpy as np
import pandas as pd
from deap import base, creator, tools, algorithms
import random

# Constants
DT = 0.02
CM0 = 0.04
MASS = 3.5
LF = 0.175
LR = 0.15

# Load CSV data
df = pd.read_csv("deep_dynamics/csv/default_params.csv")

# Function to compute state update
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

# RMSE calculation function
def calculate_rmse(params):
    """Calculate RMSE for the given parameters over all data."""
    squared_errors = []

    for _, row in df.iterrows():
        v, slip_angle, omega = row['v'], row['slip_angle'], row['omega']
        steer, accel = row['steer'], row['accel']
        
        # Predicted state
        predicted_state = compute_state_update(v, slip_angle, omega, steer, accel, params)
        actual_state = np.array([v, slip_angle, omega])
        
        squared_errors.append((actual_state - predicted_state) ** 2)
    
    rmse = np.sqrt(np.mean(squared_errors))
    return rmse

# Genetic Algorithm configuration
def configure_genetic_algorithm():
    param_bounds = [
        (0.05, 0.5),   # Iz bounds
        (20, 60),      # Df bounds
        (0.5, 5.0),    # Cf bounds
        (0.5, 5.0),    # Bf bounds
        (20, 60),      # Dr bounds
        (0.5, 5.0),    # Cr bounds
        (0.5, 5.0)     # Br bounds
    ]
    
    # DEAP configuration
    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin)

    toolbox = base.Toolbox()
    for i, (low, high) in enumerate(param_bounds):
        toolbox.register(f"attr_param_{i}", random.uniform, low, high)

    toolbox.register("individual", tools.initCycle, creator.Individual,
                     (toolbox.attr_param_0, toolbox.attr_param_1, toolbox.attr_param_2, toolbox.attr_param_3,
                      toolbox.attr_param_4, toolbox.attr_param_5, toolbox.attr_param_6), n=1)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", lambda ind: (calculate_rmse(ind),))
    toolbox.register("mate", tools.cxBlend, alpha=0.5)
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=0.1, indpb=0.2)
    toolbox.register("select", tools.selTournament, tournsize=3)

    return toolbox

# Genetic Algorithm Execution
def run_genetic_algorithm(toolbox, population_size=50, generations=100, cxpb=0.7, mutpb=0.2):
    population = toolbox.population(n=population_size)
    for gen in range(generations):
        # Apply selection, mating, and mutation
        offspring = algorithms.varAnd(population, toolbox, cxpb, mutpb)
        
        # Evaluate the individuals
        fits = map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit
        
        # Select the next generation population
        population[:] = toolbox.select(offspring, k=len(population))
        
        # Track progress
        best_ind = tools.selBest(population, k=1)[0]
        best_rmse = best_ind.fitness.values[0]
        print(f"Generation {gen + 1}/{generations}, Best RMSE: {best_rmse}")

    return population

# Main function
def main():
    toolbox = configure_genetic_algorithm()
    final_population = run_genetic_algorithm(toolbox)
    
    # Extract best individual
    best_individual = tools.selBest(final_population, k=1)[0]
    optimal_params = best_individual
    min_rmse = best_individual.fitness.values[0]
    
    # Display results
    param_names = ["Iz", "Df", "Cf", "Bf", "Dr", "Cr", "Br"]
    print("\nOptimal parameters:")
    for name, value in zip(param_names, optimal_params):
        print(f"{name}: {value}")
    print("Minimum RMSE:", min_rmse)

# Run the main function
if __name__ == "__main__":
    main()
