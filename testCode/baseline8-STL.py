import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 2. 确保数据按照日期索引
data.set_index("DATE", inplace=True)

# 3. 时序分解
# 频率设定为12表示一年12个月的季节性周期（可以根据实际情况调整）
result = seasonal_decompose(data["bal"], model="additive", period=7)

# 4. 绘制分解结果
plt.figure(figsize=(12, 8))

plt.subplot(411)
plt.plot(result.observed, label="Observed")
plt.title("Observed")
plt.grid(True)

plt.subplot(412)
plt.plot(result.trend, label="Trend", color="orange")
plt.title("Trend")
plt.grid(True)

plt.subplot(413)
plt.plot(result.seasonal, label="Seasonal", color="green")
plt.title("Seasonal")
plt.grid(True)

plt.subplot(414)
plt.plot(result.resid, label="Residual", color="red")
plt.title("Residual")
plt.grid(True)

plt.tight_layout()
plt.show()
