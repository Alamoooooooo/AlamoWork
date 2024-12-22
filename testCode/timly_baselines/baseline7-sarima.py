import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 2. 数据划分
PRED_STEP_LEN = 60
sd = "2023-09-15"
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sd:ed]

# 3. 训练SARIMA模型
# (p,d,q) 为非季节性参数, (P,D,Q,m) 为季节性参数, 这里假设月度季节性 (m=12)
# 你可以根据 ACF/PACF 图调整 p 和 q 的值
sarima_model = SARIMAX(
    train_data["bal"],
    order=(4, 0, 2),  # ARIMA部分参数
    seasonal_order=(6, 1, 3, 7),  # 季节性参数
    # order=(2, 0, 2),  # ARIMA部分参数
    # seasonal_order=(1, 1, 1, 7),  # 季节性参数
    enforce_stationarity=False,
    enforce_invertibility=False,
)

sarima_result = sarima_model.fit(disp=False)

# 4. 预测
pred_start_date = test_data.index[0]
pred_end_date = test_data.index[-1]

# 动态预测，使用训练数据训练SARIMA模型，然后对测试集进行预测
y_pred = sarima_result.predict(start=pred_start_date, end=pred_end_date)

# 5. 误差计算
mse = mean_squared_error(test_data["bal"], y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(test_data["bal"], y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 6. 绘制预测结果与真实值的对比图
plt.figure(figsize=(12, 7))
plt.plot(test_data.index, test_data["bal"], label="True Values", color="blue")
plt.plot(
    test_data.index,
    y_pred,
    label="SARIMA Predicted Values",
    color="red",
    linestyle="--",
)
plt.title("SARIMA: True vs Predicted Values")
plt.xlabel("DATE")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()
