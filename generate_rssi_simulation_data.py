import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt

# BLE 장치의 개수
num_devices = 5

# RSSI 값 범위
rssi_min = -100
rssi_max = -30

# 데이터프레임 생성
columns = ['Time'] + [f'Device_{i+1}' for i in range(num_devices)]
data = pd.DataFrame(columns=columns)

# 데이터 생성 함수
def generate_rssi_data(num_entries, interval, crowd_intervals):
    global data
    start_time = time.time()
    for i in range(num_entries):
        current_time = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time + i * interval))
        
        # 혼잡한 시간대에는 RSSI 값이 낮아짐
        if i in crowd_intervals:
            rssi_values = np.random.randint(rssi_min, rssi_min + 20, size=num_devices)
        else:
            rssi_values = np.random.randint(rssi_min + 30, rssi_max + 1, size=num_devices)
        
        row = pd.DataFrame([[current_time] + rssi_values.tolist()], columns=columns)
        data = pd.concat([data, row], ignore_index=True)

# 혼잡한 시간대 설정 (예: 10초 간격으로 혼잡이 발생)
crowd_intervals = list(range(10, 20)) + list(range(30, 40)) + list(range(50, 60))

# 60초 동안 1초 간격으로 데이터 생성
generate_rssi_data(num_entries=60, interval=1, crowd_intervals=crowd_intervals)

# CSV 파일로 저장
file_path = 'C:/Coding/Python/hanium_project/rssi_exam/rssi_data_with_pattern.csv'
data.to_csv(file_path, index=False)

# 생성된 데이터 출력
print(data)

# 파일 저장 확인 메시지 출력
print(f"CSV 파일이 성공적으로 저장되었습니다: {file_path}")