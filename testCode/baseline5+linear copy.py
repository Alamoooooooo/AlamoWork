# 随机分布拟合
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import scipy.stats as stats
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 2. 数据划分
PRED_STEP_LEN = 90
sd = "2023-10-15"
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sd:ed]

step = 15


# 特征生成函数更新：使用均值填充滞后特征的缺失值
def feature_gen(input_df):
    df = input_df.copy()

    # 日期特征
    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month
    df["Day"] = df["DATE"].dt.day
    df["DayOfWeek"] = df["DATE"].dt.dayofweek
    df["IsWeekend"] = np.where(df["DayOfWeek"] >= 5, 1, 0)

    # 滞后特征
    for i in range(1, step):  # 改为使用 14 天滞后特征
        df[f"Lag_{i}"] = df["bal"].shift(i)

    # 滚动平均特征
    df["Rolling_Mean"] = df["bal"].rolling(window=7).mean()

    # 填充缺失值为均值
    df.fillna(df.mean(), inplace=True)

    return df


# 更新后的数据处理和模型训练
train_df = feature_gen(train_data)
test_df = feature_gen(test_data)


def create_dataset(data, feature_cols, target_col):
    X = data[feature_cols]
    y = data[target_col]
    return np.array(X), np.array(y)


# 4. 构造训练集和测试集
target_col = "bal"
feature_cols = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "IsWeekend",
    # "Rolling_Mean",
] + [f"Lag_{i}" for i in range(1, step)]
X_train, y_train = create_dataset(train_df, feature_cols, target_col)
X_test, y_test = create_dataset(test_df, feature_cols, target_col)

# 6. 训练多元线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 7. 预测
y_pred = model.predict(X_test)

# 8. 误差计算
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 预测值与真实值重新排序，按照DATE_target对齐
test_df["Predicted"] = y_pred
test_df_sorted = test_df.sort_values("DATE")
print(test_df_sorted)

# 绘制预测结果与真实值的对比图
plt.figure(figsize=(12, 7))
plt.plot(
    test_df_sorted["DATE"],
    test_df_sorted["bal"],
    label="True Values",
    color="blue",
)
plt.plot(
    test_df_sorted["DATE"],
    test_df_sorted["Predicted"],
    label="Predicted Values",
    color="red",
    linestyle="--",
)
plt.title("True vs Predicted Values")
plt.xlabel("DATE")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()


residuals = y_test - y_pred
plt.figure(figsize=(12, 7))
plt.plot(test_df_sorted["DATE"], residuals, label="Residuals", color="purple")
plt.title("Residuals Over Time")
plt.xlabel("DATE")
plt.ylabel("Residuals")
plt.axhline(0, color="black", linestyle="--")
plt.legend()
plt.grid(True)
plt.show()
