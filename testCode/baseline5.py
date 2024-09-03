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
PRED_STEP_LEN = 60
sd = "2023-09-15"
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sd:ed]

step = PRED_STEP_LEN


def feature_gen(input_df):
    df = input_df.copy()

    # 特征工程：生成日期特征
    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month
    df["Day"] = df["DATE"].dt.day
    # 添加工作日和周末标记
    df["DayOfWeek"] = df["DATE"].dt.dayofweek
    df["IsWeekend"] = np.where(df["DayOfWeek"] >= 5, 1, 0)

    # 特征工程：生成滞后特征
    # （假设我们使用过去lag天的数据）
    for i in range(1, step):
        df.loc[:, f"Lag_{i}"] = df["bal"].shift(i)

    # 生成滚动平均特征（7天窗口）
    # df.loc[:, "Rolling_Mean"] = df["bal"].rolling(window=7).mean()
    return df


train_df = feature_gen(train_data)
test_df = feature_gen(test_data)
train_df.fillna(-9, inplace=True)
test_df.fillna(-9, inplace=True)


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

# 计算残差
test_df_sorted["Residual"] = test_df_sorted["bal"] - test_df_sorted["Predicted"]

# 过滤掉 NaN 或 Inf 值
df = test_df_sorted[np.isfinite(test_df_sorted["Residual"])]

# 根据工作日和休息日分别拟合正态分布
mu_weekday, std_weekday = stats.norm.fit(
    test_df_sorted[test_df_sorted["IsWeekend"] == False]["Residual"]
)
mu_weekend, std_weekend = stats.norm.fit(
    test_df_sorted[test_df_sorted["IsWeekend"] == True]["Residual"]
)

# 打印拟合的均值和标准差
print(f"Weekday Residuals: mean = {mu_weekday:.2f}, std = {std_weekday:.2f}")
print(f"Weekend Residuals: mean = {mu_weekend:.2f}, std = {std_weekend:.2f}")


# # 绘制残差的直方图
# plt.figure(figsize=(10, 7))
# plt.hist(test_df_sorted["Residual"], bins=20, color="skyblue", edgecolor="black")
# plt.title("Residuals Histogram")
# plt.xlabel("Residual")
# plt.ylabel("Frequency")
# plt.grid(True)
# plt.show()

# # 绘制残差的 Q-Q 图
# plt.figure(figsize=(10, 7))
# stats.probplot(test_df_sorted["Residual"], dist="norm", plot=plt)
# plt.title("Q-Q Plot of Residuals")
# plt.grid(True)
# plt.show()


# # 绘制拟合的正态分布曲线
# xmin, xmax = plt.xlim()
# x = np.linspace(xmin, xmax, 100)
# p = stats.norm.pdf(x, mu, std)
# plt.plot(x, p, "k", linewidth=2)
# plt.title("Residuals and Fitted Normal Distribution")
# plt.xlabel("Residual")
# plt.ylabel("Density")
# plt.grid(True)
# plt.show()


# 绘制残差的直方图和拟合的正态分布曲线
plt.figure(figsize=(12, 7))
plt.hist(
    test_df_sorted[test_df_sorted["IsWeekend"] == 0]["Residual"],
    bins=20,
    density=True,
    alpha=0.6,
    color="skyblue",
    edgecolor="black",
    label="Weekday Residuals",
)
plt.hist(
    test_df_sorted[test_df_sorted["IsWeekend"] == 1]["Residual"],
    bins=20,
    density=True,
    alpha=0.6,
    color="lightcoral",
    edgecolor="black",
    label="Weekend Residuals",
)
plt.show()

# 生成随机噪声并添加到预测值
test_df_sorted["Prediction_with_noise"] = test_df_sorted.apply(
    lambda row: (
        row["Predicted"] + np.random.normal(mu_weekday, std_weekday)
        if not row["IsWeekend"]
        else row["Predicted"] + np.random.normal(mu_weekend, std_weekend)
    ),
    axis=1,
)

# 绘制原始预测值与添加扰动后的预测值的对比
plt.figure(figsize=(12, 7))
plt.plot(
    test_df_sorted["DATE"],
    test_df_sorted["Predicted"],
    label="Original Prediction",
    color="gray",
    linestyle="--",
)
plt.plot(
    test_df_sorted["DATE"],
    test_df_sorted["Prediction_with_noise"],
    label="Prediction with Noise",
    color="blue",
)
plt.plot(
    test_df_sorted["DATE"],
    test_df_sorted["bal"],
    label="True Values",
    color="red",
)
plt.title("Original Prediction vs. Prediction with Noise")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
