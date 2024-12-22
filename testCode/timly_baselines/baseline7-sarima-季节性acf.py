import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.statespace.sarimax import SARIMAX
import itertools
from sklearn.metrics import mean_squared_error

# 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 一阶差分
data["bal_diff"] = data["bal"].diff()

# ADF检验平稳性
adf_test_diff = adfuller(data["bal_diff"].dropna())
print(f"1st Differenced Data ADF Statistic: {adf_test_diff[0]}")
print(f"1st Differenced Data p-value: {adf_test_diff[1]}")

# 绘制ACF和PACF图
plt.figure(figsize=(14, 6))
plt.subplot(121)
plot_acf(data["bal_diff"].dropna(), lags=30, ax=plt.gca())
plt.title("ACF Plot of Differenced Data")
plt.subplot(122)
plot_pacf(data["bal_diff"].dropna(), lags=30, ax=plt.gca())
plt.title("PACF Plot of Differenced Data")
plt.show()

# 定义参数范围 (p, d, q) 和 (P, D, Q, m)
p = d = q = range(0, 3)  # 非季节性参数
P = D = Q = range(0, 2)  # 季节性参数
m = [12]  # 设定为12个月的季节性周期

# 网格搜索参数组合
pdq = list(itertools.product(p, d, q))
seasonal_pdq = list(itertools.product(P, D, Q, m))

best_aic = np.inf
best_params = None
best_seasonal_params = None

for param in pdq:
    for seasonal_param in seasonal_pdq:
        try:
            mod = SARIMAX(data["bal"], order=param, seasonal_order=seasonal_param)
            results = mod.fit(disp=False)
            if results.aic < best_aic:
                best_aic = results.aic
                best_params = param
                best_seasonal_params = seasonal_param
            print(f"SARIMA{param}x{seasonal_param} - AIC:{results.aic}")
        except Exception as e:
            continue

print(f"Best SARIMA Model: {best_params} x {best_seasonal_params}, AIC: {best_aic}")
