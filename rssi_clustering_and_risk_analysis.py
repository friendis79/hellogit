import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

# 데이터 로드
data = pd.read_csv('C:/Coding/Python/hanium_project/rssi_exam/rssi_data_with_pattern.csv')

# 장치들의 RSSI 값만 사용
data = data[['Time', 'Device_1', 'Device_2', 'Device_3', 'Device_4', 'Device_5']]

# 시간 컬럼 제거 및 값 스케일링
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data.drop('Time', axis=1))

# 데이터 준비
def create_dataset(dataset, look_back=1):
    X, Y = [], []
    for i in range(len(dataset) - look_back):
        a = dataset[i:(i + look_back), :]
        X.append(a)
        Y.append(dataset[i + look_back, :])  # 모든 장치의 값을 예측
    return np.array(X), np.array(Y)

look_back = 10
X, y = create_dataset(data_scaled, look_back)

# LSTM 모델 생성 함수
def create_model(input_shape):
    model = Sequential()
    model.add(LSTM(100, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(100))
    model.add(Dense(5))  # 5개의 장치 값을 예측
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# 모델 생성 및 학습
model = create_model((look_back, 5))
model.fit(X, y, epochs=50, batch_size=32, verbose=2)

# 예측
predictions = model.predict(X)

# 실제 값으로 변환
predictions_rescaled = scaler.inverse_transform(predictions)

# 경로 손실 모델을 사용하여 RSSI 값을 거리로 변환
def rssi_to_distance(rssi, A=-40, n=3):
    return 10 ** ((A - rssi) / (10 * n))

# 실제 값과 예측 값을 거리로 변환
actual_distances = rssi_to_distance(data.iloc[look_back:, 1:].values)
predicted_distances = rssi_to_distance(predictions_rescaled)

# 밀집도 및 위험도 계산
distance_threshold = 5  # 거리 임계값 예시 (단위: meter)
density = np.mean(actual_distances < distance_threshold, axis=1)
risk = np.mean(density > 0.5)  # 임의의 기준값 0.5

# 밀집도 및 위험도 척도화 함수
def categorize_value(value):
    if value <= 0.2:
        return '매우 낮음'
    elif value <= 0.4:
        return '낮음'
    elif value <= 0.6:
        return '보통'
    elif value <= 0.8:
        return '위험'
    else:
        return '매우 위험'

# BLE 장치들 간의 거리 계산 함수
def calculate_distances(data):
    num_devices = data.shape[1]
    distances = []
    for i in range(num_devices):
        for j in range(i + 1, num_devices):
            distances.append(np.sqrt((data[:, i] - data[:, j]) ** 2))
    return np.array(distances).T

# BLE 장치들 간의 실제 거리와 예측 거리 계산
actual_inter_device_distances = calculate_distances(actual_distances)
predicted_inter_device_distances = calculate_distances(predicted_distances)

# 데이터프레임에 추가
results = pd.DataFrame(data['Time'][look_back:].reset_index(drop=True))
results['Actual_Device_1'] = data['Device_1'][look_back:].values
results['Predicted_Device_1'] = predictions_rescaled[:, 0]
results['Actual_Device_2'] = data['Device_2'][look_back:].values
results['Predicted_Device_2'] = predictions_rescaled[:, 1]
results['Actual_Device_3'] = data['Device_3'][look_back:].values
results['Predicted_Device_3'] = predictions_rescaled[:, 2]
results['Actual_Device_4'] = data['Device_4'][look_back:].values
results['Predicted_Device_4'] = predictions_rescaled[:, 3]
results['Actual_Device_5'] = data['Device_5'][look_back:].values
results['Predicted_Device_5'] = predictions_rescaled[:, 4]
results['Density'] = density
results['Risk'] = (density > 0.5).astype(int)
results['Density_Category'] = results['Density'].apply(categorize_value)
results['Risk_Category'] = results['Risk'].apply(categorize_value)

# 장치들 간의 거리 추가
distance_labels = []
for i in range(1, 6):
    for j in range(i + 1, 6):
        distance_labels.append(f'Distance_{i}_{j}')
actual_distance_df = pd.DataFrame(actual_inter_device_distances, columns=[f'Actual_{label}' for label in distance_labels])
predicted_distance_df = pd.DataFrame(predicted_inter_device_distances, columns=[f'Predicted_{label}' for label in distance_labels])

results = pd.concat([results, actual_distance_df, predicted_distance_df], axis=1)

# CSV 파일로 저장
results.to_csv('C:/Coding/Python/hanium_project/rssi_exam/results_with_actual.csv', index=False)

# BLE 장치별로 서브플롯을 만들어 그래프 피규어 5개 출력
for i in range(1, 6):
    plt.figure(figsize=(12, 8))

    plt.subplot(3, 1, 1)
    plt.plot(data['Time'][look_back:], results[f'Actual_Device_{i}'], label=f'Actual Device {i} RSSI', color='blue')
    plt.plot(data['Time'][look_back:], results[f'Predicted_Device_{i}'], label=f'Predicted Device {i} RSSI', color='red', linestyle='--')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f'Device {i} - RSSI Comparison')

    plt.subplot(3, 1, 2)
    plt.plot(data['Time'][look_back:], results['Density'], label='Density')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f'Device {i} - Density')

    plt.subplot(3, 1, 3)
    plt.plot(data['Time'][look_back:], results['Risk'], label='Risk')
    plt.xticks(rotation=45)
    plt.legend()
    plt.title(f'Device {i} - Risk')

    plt.tight_layout()
    plt.show()

# 밀집도와 위험도 출력문
avg_density = np.mean(results['Density'])
overall_risk = np.mean(results['Risk'])

avg_density_category = categorize_value(avg_density)
overall_risk_category = categorize_value(overall_risk)

print(f"Average Density: {avg_density:.2f} ({avg_density_category})")
print(f"Overall Risk: {overall_risk:.2f} ({overall_risk_category})")