import rosbag
import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import defaultdict
from scipy.interpolate import interp1d
from pyproj import Proj

# UTM 좌표계 설정 (WGS84 기준, 52 구역)
proj = Proj(proj='utm', zone=52, ellps='WGS84', preserve_units=False)

# 데이터 초기화 및 설정
def initialize_data():
    return defaultdict(list)

# 샘플링 시간 설정 (초 단위)
sampling_time = 0.01

# Bag 파일과 topic 설정
bag_path = '/mnt/d/CN7_deep_dynamics_240812.bag'
topics = {
    'gps': '/inspvax',
    'steering': '/cn7/can/avante_cn7_info',
    'accel': '/cn7/can/avante_cn7_cmd'
}

# 데이터를 처리하는 함수
def process_bag_data(bag, topics, proj):
    # 각 데이터를 저장할 딕셔너리
    data = {
        'global_x': initialize_data(),
        'global_y': initialize_data(),
        'global_yaw': initialize_data(),
        'local_vx': initialize_data(),
        'local_vy': initialize_data(),
        'local_w': initialize_data(),
        'steering': initialize_data(),
        'accel': initialize_data()
    }

    # 초기 값들
    x_offset, y_offset = None, None
    prev_yaw, prev_time = None, None
    first_gps_received = False

    start_time = bag.get_start_time()

    for topic, msg, t in tqdm(bag.read_messages(), desc="Processing Bag Data"):
        time = t.to_sec() - start_time
        if topic == topics['gps']:  # GPS Data 처리
            lat, lon = msg.latitude, msg.longitude
            x, y = proj(lon, lat)

            if not first_gps_received:
                x_offset, y_offset = x, y
                first_gps_received = True
                continue

            # 좌표 오프셋 적용
            x -= x_offset
            y -= y_offset

            # Azimuth to yaw 변환
            yaw = convert_azimuth_to_yaw(msg.azimuth)

            # 속도 계산
            vx, vy = calculate_velocity(msg.north_velocity, msg.east_velocity, yaw)

            # 데이터 저장
            store_data(data, time, x, y, yaw, vx, vy, prev_yaw, prev_time, topic)

            prev_yaw, prev_time = yaw, time

        elif topic == topics['steering']:  # Steering Data 처리
            steer_angle = np.deg2rad(msg.eps.steer_angle / 12.9)
            data['steering'][topic].append((time, steer_angle))

        elif topic == topics['accel']:  # Acceleration Data 처리
            data['accel'][topic].append((time, msg.control_command.accel))

    return data

# Azimuth to Yaw 변환 함수
def convert_azimuth_to_yaw(azimuth):
    yaw = np.pi / 2 - np.deg2rad(azimuth)
    if yaw > np.pi:
        yaw -= 2 * np.pi
    elif yaw < -np.pi:
        yaw += 2 * np.pi
    return yaw

# 속도 계산 함수
def calculate_velocity(north_velocity, east_velocity, yaw):
    vx = east_velocity * np.cos(yaw) - north_velocity * np.sin(yaw)
    vy = east_velocity * np.sin(yaw) + north_velocity * np.cos(yaw)
    return vx, vy

# 각 데이터 저장 함수
def store_data(data, time, x, y, yaw, vx, vy, prev_yaw, prev_time, topic):
    data['global_x'][topic].append((time, x))
    data['global_y'][topic].append((time, y))
    data['global_yaw'][topic].append((time, yaw))
    data['local_vx'][topic].append((time, vx))
    data['local_vy'][topic].append((time, vy))

    if prev_yaw is not None and prev_time is not None:
        delta_yaw = normalize_angle(yaw - prev_yaw)
        delta_time = time - prev_time
        angular_velocity = delta_yaw / delta_time if delta_time != 0 else 0
        data['local_w'][topic].append((time, angular_velocity))
    else:
        data['local_w'][topic].append((time, 0))

# 각도를 -pi ~ pi 범위로 정규화하는 함수
def normalize_angle(angle):
    if angle > np.pi:
        angle -= 2 * np.pi
    elif angle < -np.pi:
        angle += 2 * np.pi
    return angle

# 시간 범위 내에서 데이터를 보간하는 함수
def interpolate_data(data, new_times):
    times = [item[0] for item in data]
    values = [item[1] for item in data]
    if len(times) > 1:
        f = interp1d(times, values, kind='linear', fill_value='extrapolate')
        return f(new_times)
    else:
        return np.full_like(new_times, values[0])

# 보간 및 CSV 저장
def save_resampled_data(data, topics, min_time, max_time, sampling_time, csv_file):
    new_times = np.arange(min_time, max_time, sampling_time)

    interpolated_x = interpolate_data(data['global_x'][topics['gps']], new_times)
    interpolated_y = interpolate_data(data['global_y'][topics['gps']], new_times)
    interpolated_yaw = interpolate_data(data['global_yaw'][topics['gps']], new_times)
    interpolated_vx = interpolate_data(data['local_vx'][topics['gps']], new_times)
    interpolated_vy = interpolate_data(data['local_vy'][topics['gps']], new_times)
    interpolated_w = interpolate_data(data['local_w'][topics['gps']], new_times)
    interpolated_steering = interpolate_data(data['steering'][topics['steering']], new_times)
    interpolated_accel = interpolate_data(data['accel'][topics['accel']], new_times)

    # pandas DataFrame 생성
    df = pd.DataFrame({
        'Time': new_times,
        'X': interpolated_x,
        'Y': interpolated_y,
        'Yaw': interpolated_yaw,
        'Vx': interpolated_vx,
        'Vy': interpolated_vy,
        'W': interpolated_w,
        'Steering': interpolated_steering,
        'Accel': interpolated_accel
    })

    # CSV로 저장
    df.to_csv(csv_file, index=False)
    print(f"Resampled data saved to {csv_file}")

# 실행 함수
def main():
    # Bag 파일 열기
    bag = rosbag.Bag(bag_path)

    # 데이터 처리
    data = process_bag_data(bag, topics, proj)

    # Bag 파일 닫기
    bag.close()

    # 최소 및 최대 시간 계산
    all_times = [item[0] for item in data['global_x'][topics['gps']]]
    min_time, max_time = min(all_times), max(all_times)

    # 데이터 보간 및 저장
    csv_file = '/mnt/d/vehicle_data_resampled_4.csv'
    save_resampled_data(data, topics, min_time, max_time, sampling_time, csv_file)

# 실행
if __name__ == '__main__':
    main()
