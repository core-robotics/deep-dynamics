import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


def plot_dataset(file):
    dataset = np.load(file)
    features = dataset["features"]
    steps = len(features)
    Hz = 50

    time = np.arange(0, steps / Hz, 1 / Hz)
    vx = features[:steps, -1, 0]
    vy = features[:steps, -1, 1]
    vtheta = features[:steps, -1, 2]
    throttle = features[:steps, -1, 3]
    steering = features[:steps, -1, 4]

    # Get the dataset file name without extension
    dataset_name = os.path.splitext(os.path.basename(file))[0]

    # Create inputs/ directory if it doesn't exist
    if not os.path.exists("inputs/"):
        os.mkdir("inputs/")

    # Create a subdirectory for this dataset
    dataset_dir = os.path.join("inputs/", dataset_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)

    # Plot the inputs
    vx_fig = plt.figure()
    plt.plot(time, vx)
    plt.title("vx")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.show()

    vy_fig = plt.figure()
    plt.plot(time, vy)
    plt.title("vy")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.show()

    vtheta_fig = plt.figure()
    plt.plot(time, vtheta)
    plt.title("vtheta")
    plt.xlabel("Time (s)")
    plt.ylabel("Velocity (m/s)")
    plt.grid()
    plt.show()

    throttle_fig = plt.figure()
    plt.plot(time, throttle)
    plt.title("throttle")
    plt.xlabel("Time (s)")
    plt.ylabel("Throttle")
    plt.grid()
    plt.show()

    steering_fig = plt.figure()
    plt.plot(time, steering)
    plt.title("steering")
    plt.xlabel("Time (s)")
    plt.ylabel("rad")
    plt.grid()
    plt.show()

    # vx_error_list0 = []
    # vy_error_list0 = []
    # w_error_list0 = []
    # vx_error_list1 = []
    # vy_error_list1 = []
    # w_error_list1 = []

    # for i in range(0, steps):
    #     vx_error0, vy_error0, w_error0 = get_estimate_error(
    #         vx[i], vy[i], vtheta[i], throttle[i], steering[i], sys_param0
    #     )
    #     vx_error1, vy_error1, w_error1 = get_estimate_error(
    #         vx[i], vy[i], vtheta[i], throttle[i], steering[i], sys_param1
    #     )

    #     vx_error_list0.append(vx_error0)
    #     vx_error_list1.append(vx_error1)
    #     vy_error_list0.append(vy_error0)
    #     vy_error_list1.append(vy_error1)
    #     w_error_list0.append(w_error0)
    #     w_error_list1.append(w_error1)

    # # Create the combined lists for each error type
    # vx_error_combined = vx_error_list0 + vx_error_list1
    # vy_error_combined = vy_error_list0 + vy_error_list1
    # w_error_combined = w_error_list0 + w_error_list1

    # vx_rmse0 = get_rmse(vx_error_list0)
    # vy_rmse0 = get_rmse(vy_error_list0)
    # w_rmse0 = get_rmse(w_error_list0)

    # vx_rmse1 = get_rmse(vx_error_list1)
    # vy_rmse1 = get_rmse(vy_error_list1)
    # w_rmse1 = get_rmse(w_error_list1)

    # # Create the data_type list to differentiate between Original and DeepDynamics
    # data_type = ["Original"] * len(vx_error_list0) + ["DeepDynamics"] * len(
    #     vx_error_list1
    # )

    # # Create DataFrame with the combined data
    # df = pd.DataFrame(
    #     {
    #         "vx_error": vx_error_combined,
    #         "vy_error": vy_error_combined,
    #         "w_error": w_error_combined,
    #         "data_type": data_type,
    #     }
    # )

    # # Melt the DataFrame to make it easier to plot
    # df_melted = pd.melt(
    #     df, id_vars=["data_type"], var_name="error_type", value_name="error_value"
    # )

    # # Set plot style
    # sns.set_style("whitegrid")

    # # Plot the boxplot with hue for data_type
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x="error_type", y="error_value", hue="data_type", data=df_melted)

    # # Set plot labels and title
    # plt.title("Error Comparison between Original and DeepDynamics")

    # plt.xticks(
    #     [0, 1, 2],
    #     [
    #         f"vx\n{vx_rmse0:.4f},{vx_rmse1:.4f}",
    #         f"vy\n{vy_rmse0:.4f},{vy_rmse1:.4f}",
    #         f"w\n{w_rmse0:.4f},{w_rmse1:.4f}",
    #     ],
    # )

    # # Show the plot
    # plt.show()


if __name__ == "__main__":
    import argparse, argcomplete

    parser = argparse.ArgumentParser(description="Visualize a dataset.")
    parser.add_argument("dataset_file", type=str, help="Dataset to plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict: dict = vars(args)
    plot_dataset(argdict["dataset_file"])
