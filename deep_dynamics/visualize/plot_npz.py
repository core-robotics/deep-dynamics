import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os


vehicle_spec = {
    "mass": 3.5,
    "lf": 0.17145,
    "lr": 0.17145
    }
    
sys_param0 = {
    "Bf": 1.5,
    "Cf": 1.5,
    "Df": 30.0,
    "Br": 1.5,
    "Cr": 1.5,
    "Dr": 30.0,
    "Iz": 0.04712
}

sys_param1 = {
    "Bf": 2.4450,
    "Cf": 1.1975,
    "Df": 26.1542,
    "Br": 2.8304,
    "Cr": 1.1916,
    "Dr": 26.6161,
    "Iz": 0.06491
}

def differential_equation(vehicle_specs, sys_param_dict, dt, vx,vy,w, throttle, steering):
    # 각도 계산
    alphaf = steering - np.arctan2(vehicle_specs["lf"] *w + vy, vx)
    alphar = np.arctan2(vehicle_specs["lr"] * w - vy, vx)

    # 가속도 및 힘 계산
    # F = vehicle_specs["mass"] * throttle
    Ffy = sys_param_dict["Df"] * np.sin(sys_param_dict["Cf"] * np.arctan(sys_param_dict["Bf"] * alphaf))
    Fry = sys_param_dict["Dr"] * np.sin(sys_param_dict["Cr"] * np.arctan(sys_param_dict["Br"] * alphar))
    
    slip_angle=np.atan2(vy,vx)
    Fx= vehicle_specs["mass"] * throttle*np.cos(slip_angle)

    # 상태 변화율 
    vx_dot = (1 / vehicle_specs["mass"]) * (Fx - Ffy * np.sin(steering)) +vy * w
    vy_dot = (1 / vehicle_specs["mass"]) * (Fry + Ffy * np.cos(steering)) - vx * w
    w_dot = (1 / sys_param_dict["Iz"]) * (Ffy * vehicle_specs["lf"] * np.cos(steering) - Fry * vehicle_specs["lr"])

    vx_next = vx + vx_dot * dt
    vy_next = vy + vy_dot * dt
    w_next = w + w_dot * dt
    
    return vx_next, vy_next, w_next

def get_estimate_error(vx_gt, vy_gt, w_gt, throttle, steering, param):
    vx_predict, vy_predict, w_predict = differential_equation(vehicle_spec, param, 0.025, vx_gt, vy_gt, w_gt, throttle, steering)
    vx_error = vx_gt - vx_predict
    vy_error = vy_gt - vy_predict
    w_error = w_gt - w_predict
    return vx_error, vy_error, w_error

def get_rmse(error_list):
    return np.sqrt(np.mean(np.square(error_list)))

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

    
    # Get the dataset file name without extension
    dataset_name = os.path.splitext(os.path.basename(file))[0]
    
    # Create inputs/ directory if it doesn't exist
    if not os.path.exists("inputs/"):
        os.mkdir("inputs/")
    
    # Create a subdirectory for this dataset
    dataset_dir = os.path.join("inputs/", dataset_name)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    
    vx_error_list0 = []
    vy_error_list0 = []
    w_error_list0 = []
    vx_error_list1 = []
    vy_error_list1 = []
    w_error_list1 = []
    
    
    for i in range(0,steps):
        vx_error0, vy_error0, w_error0 = get_estimate_error(vx[i], vy[i], vtheta[i], throttle[i], steering[i], sys_param0)
        vx_error1, vy_error1, w_error1 = get_estimate_error(vx[i], vy[i], vtheta[i], throttle[i], steering[i], sys_param1)
        
        vx_error_list0.append(vx_error0)
        vx_error_list1.append(vx_error1)
        vy_error_list0.append(vy_error0)
        vy_error_list1.append(vy_error1)
        w_error_list0.append(w_error0)
        w_error_list1.append(w_error1)
        
    # Create the combined lists for each error type
    vx_error_combined = vx_error_list0 + vx_error_list1
    vy_error_combined = vy_error_list0 + vy_error_list1
    w_error_combined = w_error_list0 + w_error_list1

    vx_rmse0 = get_rmse(vx_error_list0)
    vy_rmse0 = get_rmse(vy_error_list0)
    w_rmse0 = get_rmse(w_error_list0)
    
    vx_rmse1 = get_rmse(vx_error_list1)
    vy_rmse1 = get_rmse(vy_error_list1)
    w_rmse1 = get_rmse(w_error_list1)
    
    # Create the data_type list to differentiate between Original and DeepDynamics
    data_type = ['Original'] * len(vx_error_list0) + ['DeepDynamics'] * len(vx_error_list1)

    # Create DataFrame with the combined data
    df = pd.DataFrame({
        'vx_error': vx_error_combined,
        'vy_error': vy_error_combined,
        'w_error': w_error_combined,
        'data_type': data_type
    })

    # Melt the DataFrame to make it easier to plot
    df_melted = pd.melt(df, id_vars=['data_type'], var_name='error_type', value_name='error_value')

    # Set plot style
    sns.set_style("whitegrid")

    # Plot the boxplot with hue for data_type
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='error_type', y='error_value', hue='data_type', data=df_melted)

    # Set plot labels and title
    plt.title("Error Comparison between Original and DeepDynamics")

    plt.xticks(
        [0,1,2],
        [f'vx\n{vx_rmse0:.4f},{vx_rmse1:.4f}', f'vy\n{vy_rmse0:.4f},{vy_rmse1:.4f}', f'w\n{w_rmse0:.4f},{w_rmse1:.4f}']
        
    )

    # Show the plot
    plt.show()
    

    
   
    
    
    
    # data=np.array([vx_error_list0, vx_error_list1, vy_error_list0, vy_error_list1, w_error_list0, w_error_list1])
    
    # # 박스 플롯을 그리기
    # boxplot = plt.boxplot(data.T, tick_labels=["vx_error0", "vx_error1", "vy_error0", "vy_error1", "w_error0", "w_error1"])

    # # 각 box plot의 상, 하한 값을 텍스트로 표시
    # for i, line in enumerate(boxplot['whiskers']):
    #     # 상한 및 하한 값의 인덱스는 짝수(하한), 홀수(상한)로 저장됨
    #     if i % 2 == 0:
    #         ymin = line.get_ydata()[1]  # 하한
    #         plt.text(i // 2+1, ymin-0.1, f'{ymin:.2f}', ha='center', va='top')
    #     else:
    #         ymax = line.get_ydata()[1]  # 상한
    #         plt.text(i // 2+1 , ymax+0.1, f'{ymax:.2f}', ha='center', va='bottom')
            
    # for i, line in enumerate(boxplot['medians']):
    #     median = line.get_ydata()[0]
    #     plt.text(i+1, median, f'{median:.2f}', ha='center', va='bottom')
        
    # for i, box in enumerate(boxplot['boxes']):
    #     q1 = box.get_ydata()[0]  # Q1 값은 박스의 아래쪽 모서리
    #     q3 = box.get_ydata()[2]  # Q3 값은 박스의 위쪽 모서리
    #     plt.text(i + 1, q1 , f'{q1:.2f}', ha='center', va='top')  # Q1 값
    #     plt.text(i + 1, q3 , f'{q3:.2f}', ha='center', va='bottom')  # Q3 값
    
    
    # plt.xlim([0, 7])
    # plt.ylim([-1, 1])
    # # plt.savefig(os.path.join(dataset_dir, 'error_boxplot.png'))
    
    # # plt.close()
    # plt.show()
        
    # Plot and save each graph in the specific folder
    # Plot for vx
    # plt.figure()
    # plt.plot(time, vx_error_list0, label="$car0$ ($m/s$)")
    # plt.plot(time, vx_error_list1, label="$car1$ ($m/s$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(min(vx_error_list0), min(vx_error_list1)), max(max(vx_error_list0), max(vx_error_list1))])
    # plt.xlabel("Time (s)")
    # plt.ylabel("$v_{x error}$ ($m/s$)")  # 이중 하첨자 문제 해결
    # plt.title("Velocity in x-direction")
    # plt.legend()
    # plt.savefig(os.path.join(dataset_dir, 'vx_error_plot.png'))
    # plt.close()

    # # Plot for vy
    # plt.figure()
    # plt.plot(time, vy_error_list0, label="$car0$ ($m/s$)")
    # plt.plot(time, vy_error_list1, label="$car1$ ($m/s$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(min(vy_error_list0), min(vy_error_list1)), max(max(vy_error_list0), max(vy_error_list1))])
    # plt.xlabel("Time (s)")
    # plt.ylabel("$v_{y error}$ ($m/s$)")  # 이중 하첨자 문제 해결
    # plt.title("Velocity in y-direction")
    # plt.legend()
    # plt.savefig(os.path.join(dataset_dir, 'vy_error_plot.png'))
    # plt.close()

    # # Plot for vtheta (Angular velocity)
    # plt.figure()
    # plt.plot(time, w_error_list0, label="$car0$ ($rad/s$)")
    # plt.plot(time, w_error_list1, label="$car1$ ($rad/s$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(min(w_error_list0), min(w_error_list1)), max(max(w_error_list0), max(w_error_list1))])
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angular velocity error ($rad/s$)")  # 수식 아닌 일반 텍스트로 사용
    # plt.title("Angular Velocity")
    # plt.legend()
    # plt.savefig(os.path.join(dataset_dir, 'vtheta_error_plot.png'))
    # plt.close()
        
        
        
    
    

        
      
    
    
    
    
    
    # # Plot and save each graph in the specific folder
    # # Plot for vx
    # plt.figure()
    # plt.plot(time, vx, label="$v_x$ ($m/s$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(vx), max(vx)])
    # plt.xlabel("Time (s)")
    # plt.ylabel("$v_x$ ($m/s$)")
    # plt.title("Velocity in x-direction")
    # plt.savefig(os.path.join(dataset_dir, 'vx_plot.png'))
    # plt.close()

    # # Plot for vy
    # plt.figure()
    # plt.plot(time, vy, label="$v_y$ ($m/s$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(vy), max(vy)])
    # plt.xlabel("Time (s)")
    # plt.ylabel("$v_y$ ($m/s$)")
    # plt.title("Velocity in y-direction")
    # plt.savefig(os.path.join(dataset_dir, 'vy_plot.png'))
    # plt.close()

    # # Plot for vtheta
    # plt.figure()
    # plt.plot(time, vtheta, label="$\omega$ ($rad/s$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(vtheta), max(vtheta)])
    # plt.xlabel("Time (s)")
    # plt.ylabel("Angular velocity ($rad/s$)")
    # plt.title("Angular Velocity")
    # plt.savefig(os.path.join(dataset_dir, 'vtheta_plot.png'))
    # plt.close()

    # # Plot for throttle
    # plt.figure()
    # plt.plot(time, throttle, label="Throttle (%)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(throttle), max(throttle)])
    # plt.xlabel("Time (s)")
    # plt.ylabel("Throttle (%)")
    # plt.title("Throttle Input")
    # plt.savefig(os.path.join(dataset_dir, 'throttle_plot.png'))
    # plt.close()

    # # Plot for steering
    # plt.figure()
    # plt.plot(time, steering, label="Steering Angle ($rad$)")
    # plt.xlim([time[0], time[-1]])
    # plt.ylim([min(steering), max(steering)])
    # plt.xlabel("Time (s)")
    # plt.ylabel("Steering Angle ($rad$)")
    # plt.title("Steering Input")
    # plt.savefig(os.path.join(dataset_dir, 'steering_plot.png'))
    # plt.close()

if __name__ == "__main__":
    import argparse, argcomplete
    parser = argparse.ArgumentParser(description="Visualize a dataset.")
    parser.add_argument("dataset_file", type=str, help="Dataset to plot")
    argcomplete.autocomplete(parser)
    args = parser.parse_args()
    argdict: dict = vars(args)
    plot_dataset(argdict["dataset_file"])
