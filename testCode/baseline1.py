import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 禁用 SettingWithCopyWarning
pd.options.mode.chained_assignment = None  # 关闭警告


def feature_gen(input_df):
    df = input_df.copy()
    df["Date"] = pd.to_datetime(df["Date"])

    # 特征工程：生成滞后特征和日期特征
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Day"] = df["Date"].dt.day
    # 添加工作日和周末标记
    df["DayOfWeek"] = df["Date"].dt.dayofweek
    df["IsWeekend"] = df["DayOfWeek"] >= 5
    # 生成滞后特征
    # （假设我们使用过去lag天的数据）
    for i in range(1, 30):
        df.loc[:, f"Lag_{i}"] = df["FinalVolume"].shift(i)

    # 生成滚动平均特征（7天窗口）
    df.loc[:, "Rolling_Mean"] = df["FinalVolume"].rolling(window=7).mean()
    return df


df_ori = pd.read_csv("data.csv")[["Date", "FinalVolume"]]

# 按指定的日期进行数据划分
split_date = "2023-12-01"
train_df = df_ori[df_ori["Date"] < split_date]
test_df = df_ori[df_ori["Date"] >= split_date]

# 分别在训练集和测试集上生成特征
train_df = feature_gen(train_df)
test_df = feature_gen(test_df)

# 删除因滚动计算而产生的NA值
# train_df.dropna(inplace=True)
# test_df.dropna(inplace=True)

# 填充因滚动计算而产生的NA值
train_df.fillna(-1, inplace=True)
test_df.fillna(-1, inplace=True)

fea_cols = [
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "Rolling_Mean",
] + [f"Lag_{i}" for i in range(1, 30)]

X_train = train_df[fea_cols]
y_train = train_df["FinalVolume"]
X_test = test_df[fea_cols]
y_test = test_df["FinalVolume"]

print(train_df)
print(test_df)

# 创建线性回归模型
model = LinearRegression()
model.fit(X_train, y_train)

# 预测
y_pred = model.predict(X_test)

# 评估指标
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 绘制预测结果与真实值的对比图
plt.figure(figsize=(14, 7))
plt.plot(test_df["Date"], y_test, label="True Values", color="blue", marker="o")
plt.plot(
    test_df["Date"],
    y_pred,
    label="Predicted Values",
    color="red",
    linestyle="--",
    marker="x",
)
plt.title("True vs Predicted Values")
plt.xlabel("Date")
plt.ylabel("Trading Volume")
plt.legend()
plt.grid(True)
plt.show()
