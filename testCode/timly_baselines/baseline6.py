import numpy as np
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegressicdon
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. 加载数据
df = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
df.columns = ["Date", "Value"]
df["Date"] = pd.to_datetime(df["Date"])
df.set_index("Date", inplace=True)


# 创建星期几和是否为周末的虚拟变量
df["DayOfWeek"] = df.index.dayofweek
df["IsWeekend"] = (df["DayOfWeek"] >= 5).astype(int)

# 创建星期几的虚拟变量（排除一个类别，以避免虚拟变量陷阱）
df = pd.get_dummies(df, columns=["DayOfWeek"], drop_first=True)

# 打印列名以检查虚拟变量列名
print("Columns after get_dummies:\n", df.columns)

# 生成时间趋势变量
df["Trend"] = np.arange(len(df))

# 切分数据集
pred_len = 60  # 预测集的长度
train_df = df.iloc[:-pred_len]  # 训练集
test_df = df.iloc[-pred_len:]  # 测试集

# 获取虚拟变量的列名
dummy_columns = [col for col in df.columns if col.startswith("DayOfWeek_")]

# 训练集的预测变量和目标变量
X_train = train_df[["Trend", "IsWeekend"] + dummy_columns]
y_train = train_df["Value"]

# 测试集的预测变量
X_test = test_df[["Trend", "IsWeekend"] + dummy_columns]

# 添加常数项（截距）
X_train = sm.add_constant(X_train)
X_test = sm.add_constant(X_test)

# 构建并拟合回归模型
model = sm.OLS(y_train, X_train).fit()

# 训练集的预测值
train_df["Predicted"] = model.predict(X_train)

# 测试集的预测值
test_df["Predicted"] = model.predict(X_test)

# 输出回归结果
print(model.summary())

# 可视化结果
plt.figure(figsize=(12, 7))
plt.plot(train_df.index, train_df["Value"], label="Train Actual")
plt.plot(
    train_df.index,
    train_df["Predicted"],
    label="Train Predicted",
    # linestyle="--",
)
plt.plot(test_df.index, test_df["Value"], label="Test Actual", color="green")
plt.plot(
    test_df.index,
    test_df["Predicted"],
    label="Test Predicted",
    # linestyle="--",
    color="red",
)
plt.title("Time Series Prediction with Train-Test Split")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# 可视化结果
plt.figure(figsize=(12, 7))
plt.plot(test_df.index, test_df["Value"], label="Test Actual", color="green")
plt.plot(
    test_df.index,
    test_df["Predicted"],
    label="Test Predicted",
    # linestyle="--",
    color="red",
)
plt.title("Time Series Prediction with Train-Test Split")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
