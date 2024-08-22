import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 차량 및 시스템 파라미터 정의
vehicle_specs = {
    "lf": 1.5,  # 전륜 축 길이 (예시 값, 실제 값 입력 필요)
    "lr": 1.5,  # 후륜 축 길이 (예시 값, 실제 값 입력 필요)
    "mass": 1500  # 차량 질량 (kg)
}

sys_param_dict = {
    "Cm1": 1.0,  # Longitudinal force constant (예시 값)
    "Df": 3000.0, "Cf": 1.2, "Bf": 1.0,  # Front tire parameters
    "Dr": 3000.0, "Cr": 1.2, "Br": 1.0  # Rear tire parameters
}

# CSV 파일 읽기
file_path = '/home/a/deep-dynamics/deep_dynamics/data/CN7_1.csv'
df = pd.read_csv(file_path)

# 상태 변수 정의
Vx = df['Vx'].values
Vy = df['Vy'].values
Yaw_rate = df['W'].values
Steering = df['Steering'].values
Accel = df['Accel'].values
Time = df['Time'].values

# Trajectory에 사용할 X, Y 좌표
X = df['X'].values
Y = df['Y'].values

# 각도를 라디안으로 변환 (필요시)
Steering = np.deg2rad(Steering)

# 앞바퀴와 뒷바퀴의 슬립 각도 계산
a_f = Steering - np.arctan2(vehicle_specs["lf"] * Yaw_rate + Vy, np.abs(Vx))
a_r = np.arctan2(vehicle_specs["lr"] * Yaw_rate - Vy, np.abs(Vx))

# 앞바퀴와 뒷바퀴의 횡력 계산
Ffy = sys_param_dict["Df"] * np.sin(sys_param_dict["Cf"] * np.arctan(sys_param_dict["Bf"] * a_f))
Fry = sys_param_dict["Dr"] * np.sin(sys_param_dict["Cr"] * np.arctan(sys_param_dict["Br"] * a_r))

# 시간에 따른 슬립각과 횡력, 궤적 그래프 그리기
plt.figure(figsize=(12, 8))

# 1. Slip Angles Plot
plt.subplot(3, 1, 1)
plt.plot(Time, a_f, label='Front Slip Angle (a_f)')
plt.plot(Time, a_r, label='Rear Slip Angle (a_r)')
plt.title('Slip Angles Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Slip Angle (rad)')
plt.legend()

# 2. Lateral Forces Plot
plt.subplot(3, 1, 2)
plt.plot(Time, Ffy, label='Front Lateral Force (F_fy)')
plt.plot(Time, Fry, label='Rear Lateral Force (F_ry)')
plt.title('Lateral Forces Over Time')
plt.xlabel('Time (s)')
plt.ylabel('Lateral Force (N)')
plt.legend()

# 3. Trajectory Plot (X, Y)
plt.subplot(3, 1, 3)
plt.plot(X, Y, label='Trajectory (X vs Y)')
plt.title('Vehicle Trajectory')
plt.xlabel('X Position (m)')
plt.ylabel('Y Position (m)')
plt.legend()

plt.tight_layout()
plt.show()
