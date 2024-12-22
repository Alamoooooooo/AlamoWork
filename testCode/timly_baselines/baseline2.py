import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 假设 df 包含了历史的日期和 FinalVolume
df = pd.read_csv("data.csv")

# 定义窗口大小
history_size = 7
forecast_horizon = 30

# 获取最后的历史数据
df["Date"] = pd.to_datetime(df["Date"])
last_known_date = df["Date"].max()
last_known_data = df[df["Date"] >= last_known_date - pd.Timedelta(days=history_size)]

# 确保最后7天数据的完整性
last_known_data = last_known_data.tail(history_size)

# 提取最后7天的数据作为输入特征
X_input = last_known_data["FinalVolume"].values.reshape(1, -1)  # 1x7 的数组

# 初始化线性回归模型
model = LinearRegression()

# 使用最近的7天数据作为输入，构建30个目标输出
# 在这个示例中，我们假设已经有训练好的模型，或假设模型是使用历史数据训练过的。
# 在实际场景下，模型应该是经过适当的训练的。
# 这里直接以最近7天的数据作为输入特征，并输出未来30天的数据。
# 模拟训练数据
X_train = []
y_train = []
for i in range(len(df) - history_size - forecast_horizon):
    X_train.append(df["FinalVolume"].iloc[i : i + history_size].values)
    y_train.append(
        df["FinalVolume"]
        .iloc[i + history_size : i + history_size + forecast_horizon]
        .values
    )

X_train = np.array(X_train)
y_train = np.array(y_train)


# 训练模型
model.fit(X_train, y_train)

# 6. 使用最近7天的数据预测未来30天
# 构建预测输入
X_pred = np.tile(X_input, (forecast_horizon, 1))  # 重复7天数据，形成 (30, 7) 的矩阵

# 进行预测
y_pred = model.predict(X_pred)

# 因为我们期望每次预测未来一天，因此需要对结果取平均或进行适当的处理
predicted_volumes = y_pred.mean(axis=0)

# 构建预测的日期
forecast_dates = pd.date_range(
    start=last_known_date + pd.Timedelta(days=1), periods=forecast_horizon
)
forecast_df = pd.DataFrame(
    {"Date": forecast_dates, "PredictedVolume": predicted_volumes}
)

# 输出预测结果
print(forecast_df)

# 可视化预测结果
import matplotlib.pyplot as plt

plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["FinalVolume"], label="Historical Data", color="blue")
plt.plot(
    forecast_df["Date"],
    forecast_df["PredictedVolume"],
    label="Predicted Data",
    color="red",
    linestyle="--",
)
plt.title("Future 30-Day Prediction")
plt.xlabel("Date")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()
