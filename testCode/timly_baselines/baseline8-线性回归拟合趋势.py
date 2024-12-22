import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 确保数据按照日期排序
data.set_index("DATE", inplace=True)

# 设定预测步长，分割训练集和测试集
PRED_STEP_LEN = 60
sd = "2023-09-15"
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN - 1)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sd:ed]


# 1. 使用时序分解
result = seasonal_decompose(train_data["bal"], model="additive", period=12)

# 2. 对趋势进行线性回归外推
trend = result.trend.dropna()  # 去掉NaN值

# 生成训练集
X_train = np.arange(len(trend)).reshape(-1, 1)
y_train = trend.values

# 线性回归模型
model = LinearRegression().fit(X_train, y_train)

# 预测未来趋势（未来60天）
future_steps = PRED_STEP_LEN
X_future = np.arange(len(trend), len(trend) + future_steps).reshape(-1, 1)
trend_forecast = model.predict(X_future)


# 3. 使用季节性分量外推
seasonal = result.seasonal[-12:]  # 假设季节性周期为12
seasonal_forecast = np.tile(seasonal, int(np.ceil(future_steps / 12)))[:future_steps]


# 4. 将趋势和季节性分量相加得到最终预测值
final_forecast_trend = trend_forecast + seasonal_forecast


# 绘制趋势预测结果
plt.figure(figsize=(10, 6))
plt.plot(
    test_data.index, final_forecast_trend, label="Trend Prediction", color="orange"
)
plt.plot(test_data.index, test_data["bal"], label="True Values", color="blue")
plt.title("Trend Prediction vs True Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# 5. 误差计算
mse = mean_squared_error(test_data["bal"], final_forecast_trend)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data["bal"], final_forecast_trend)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
print(f"Mean Absolute Error: {mae}")
