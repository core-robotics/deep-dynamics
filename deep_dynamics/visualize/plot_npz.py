import numpy as np
import matplotlib.pyplot as plt
import os

def plot_dataset(file):
    dataset = np.load(file)
    features = dataset['features']
    steps=len(features)
    Hz=40
    
    time= np.arange(0,steps/Hz,1/Hz)
    vx = features[:steps, -1, 0]
    vy = features[:steps, -1, 1]
    vtheta = features[:steps, -1, 2]
    throttle = features[:steps, -1, 3]
    steering = features[:steps, -1, 4]
    # vx = features[:, -1, 0]
    # vy = features[:, -1, 1]
    # vtheta = features[:, -1, 2]
    # throttle = features[:, -1, 3]
    # steering = features[:, -1, 4]
    
    # Get the dataset file name without extension
    dataset_name = os.path.splitext(os.path.basename(file))[0]
    
    # Create inputs/ directory if it doesn't exist
    if not os.path.exists("inputs/"):
        os.mkdir("inputs/")
    
    # Create a subdirectory for this dataset
    dataset_dir = os.path.join("inputs/", dataset_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    # Plot and save each graph in the specific folder
    # Plot for vx
    plt.figure()
    plt.plot(time, vx, label="$v_x$ ($m/s$)")
    plt.xlim([time[0], time[-1]])
    plt.ylim([min(vx), max(vx)])
    plt.xlabel("Time (s)")
    plt.ylabel("$v_x$ ($m/s$)")
    plt.title("Velocity in x-direction")
    plt.savefig(os.path.join(dataset_dir, 'vx_plot.png'))
    plt.close()

    # Plot for vy
    plt.figure()
    plt.plot(time, vy, label="$v_y$ ($m/s$)")
    plt.xlim([time[0], time[-1]])
    plt.ylim([min(vy), max(vy)])
    plt.xlabel("Time (s)")
    plt.ylabel("$v_y$ ($m/s$)")
    plt.title("Velocity in y-direction")
    plt.savefig(os.path.join(dataset_dir, 'vy_plot.png'))
    plt.close()

    # Plot for vtheta
    plt.figure()
    plt.plot(time, vtheta, label="$\omega$ ($rad/s$)")
    plt.xlim([time[0], time[-1]])
    plt.ylim([min(vtheta), max(vtheta)])
    plt.xlabel("Time (s)")
    plt.ylabel("Angular velocity ($rad/s$)")
    plt.title("Angular Velocity")
    plt.savefig(os.path.join(dataset_dir, 'vtheta_plot.png'))
    plt.close()

    # Plot for throttle
    plt.figure()
    plt.plot(time, throttle, label="Throttle (%)")
    plt.xlim([time[0], time[-1]])
    plt.ylim([min(throttle), max(throttle)])
    plt.xlabel("Time (s)")
    plt.ylabel("Throttle (%)")
    plt.title("Throttle Input")
    plt.savefig(os.path.join(dataset_dir, 'throttle_plot.png'))
    plt.close()

    # Plot for steering
    plt.figure()
    plt.plot(time, steering, label="Steering Angle ($rad$)")
    plt.xlim([time[0], time[-1]])
    plt.ylim([min(steering), max(steering)])
    plt.xlabel("Time (s)")
    plt.ylabel("Steering Angle ($rad$)")
    plt.title("Steering Input")
    plt.savefig(os.path.join(dataset_dir, 'steering_plot.png'))
    plt.close()

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Visualize a dataset.")
    parser.add_argument("dataset_file", type=str, help="Dataset to plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict: dict = vars(args)
    plot_dataset(argdict["dataset_file"])
