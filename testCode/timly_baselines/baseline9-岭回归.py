import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
from datetime import timedelta

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 确保数据按照日期排序
data.set_index("DATE", inplace=True)

# 2. 数据划分
PRED_STEP_LEN = 60
sd = "2023-09-15"
sdd = (pd.to_datetime(sd) + timedelta(days=1)).strftime("%Y-%m-%d")
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sdd:ed]

step = PRED_STEP_LEN


# 3. 特征工程
def feature_gen(input_df):
    df = input_df.copy()

    # 生成日期特征
    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month
    df["Day"] = df["DATE"].dt.day
    df["DayOfWeek"] = df["DATE"].dt.dayofweek
    df["IsWeekend"] = np.where(df["DayOfWeek"] >= 5, 1, 0)

    # 滞后特征
    for i in range(1, step):
        df.loc[:, f"Lag_{i}"] = df["bal"].shift(i)

    return df


train_df = feature_gen(train_data)
test_df = feature_gen(test_data)

# 填充缺失值
train_df.fillna(-9, inplace=True)
test_df.fillna(-9, inplace=True)

# 4. 构造训练集和测试集
target_col = "bal"
feature_cols = ["Year", "Month", "Day", "DayOfWeek", "IsWeekend"] + [
    f"Lag_{i}" for i in range(1, step)
]


def create_dataset(data, feature_cols, target_col):
    X = data[feature_cols]
    y = data[target_col]
    return np.array(X), np.array(y)


X_train, y_train = create_dataset(train_df, feature_cols, target_col)
X_test, y_test = create_dataset(test_df, feature_cols, target_col)

# 5. 训练岭回归模型
ridge_model = Ridge(alpha=1.0)  # alpha是正则化强度的参数
ridge_model.fit(X_train, y_train)

# 6. 预测
y_pred = ridge_model.predict(X_test)

# 7. 误差计算
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
mae = mean_absolute_error(y_test, y_pred)

print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"Root Mean Squared Error (RMSE): {rmse:.2f}")
print(f"Mean Absolute Error (MAE): {mae:.2f}")

# 8. 结果可视化
test_df["Predicted"] = y_pred
test_df_sorted = test_df.sort_values("DATE")

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
    label="Predicted Values (Ridge Regression)",
    color="red",
    linestyle="--",
)
plt.title("True vs Predicted Values - Ridge Regression")
plt.xlabel("DATE")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
