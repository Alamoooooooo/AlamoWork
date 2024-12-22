import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 2. 绘制原始数据
plt.figure(figsize=(10, 6))
plt.plot(data["DATE"], data["bal"], label="Original Data")
plt.title("Original Time Series Data")
plt.grid(True)
plt.show()

# 3. ADF 检验原始数据的平稳性
adf_test_orig = adfuller(data["bal"])
print(f"Original Data ADF Statistic: {adf_test_orig[0]}")
print(f"Original Data p-value: {adf_test_orig[1]}")

# 4. 进行一阶差分以去除趋势
data["bal_diff"] = data["bal"].diff()

# 5. 绘制一阶差分后的数据
plt.figure(figsize=(10, 6))
plt.plot(data["DATE"], data["bal_diff"], label="1st Differenced Data", color="orange")
plt.title("First Order Differenced Time Series Data")
plt.grid(True)
plt.show()

# 6. ADF 检验一阶差分数据的平稳性
adf_test_diff = adfuller(data["bal_diff"].dropna())
print(f"1st Differenced Data ADF Statistic: {adf_test_diff[0]}")
print(f"1st Differenced Data p-value: {adf_test_diff[1]}")

# 7. 绘制差分后的 ACF 和 PACF 图
plt.figure(figsize=(14, 6))
plt.subplot(121)
plot_acf(data["bal_diff"].dropna(), lags=30, ax=plt.gca())
plt.title("ACF of Differenced Data")
plt.subplot(122)
plot_pacf(data["bal_diff"].dropna(), lags=30, ax=plt.gca())
plt.title("PACF of Differenced Data")
plt.show()
