from sklearn.linear_model import Ridge, Lasso
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
import scipy.stats as stats

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


# 1. 数据标准化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 2. Ridge 回归模型训练
ridge_model = Ridge(alpha=1.0)  # alpha 是正则化强度的参数，可以调整
ridge_model.fit(X_train_scaled, y_train)

# 3. Ridge 回归模型预测
y_pred_ridge = ridge_model.predict(X_test_scaled)

# 4. Lasso 回归模型训练
lasso_model = Lasso(alpha=0.1)  # alpha 是正则化强度的参数，可以调整
lasso_model.fit(X_train_scaled, y_train)

# 5. Lasso 回归模型预测
y_pred_lasso = lasso_model.predict(X_test_scaled)

# 6. 误差计算
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
rmse_ridge = np.sqrt(mse_ridge)
mae_ridge = mean_absolute_error(y_test, y_pred_ridge)

mse_lasso = mean_squared_error(y_test, y_pred_lasso)
rmse_lasso = np.sqrt(mse_lasso)
mae_lasso = mean_absolute_error(y_test, y_pred_lasso)

print(f"Ridge - Mean Squared Error (MSE): {mse_ridge:.2f}")
print(f"Ridge - Root Mean Squared Error (RMSE): {rmse_ridge:.2f}")
print(f"Ridge - Mean Absolute Error (MAE): {mae_ridge:.2f}")

print(f"Lasso - Mean Squared Error (MSE): {mse_lasso:.2f}")
print(f"Lasso - Root Mean Squared Error (RMSE): {rmse_lasso:.2f}")
print(f"Lasso - Mean Absolute Error (MAE): {mae_lasso:.2f}")

# 绘制预测结果与真实值的对比图（Ridge 回归）
plt.figure(figsize=(12, 7))
plt.plot(y_test, label="True Values", color="blue")
plt.plot(
    y_pred_ridge,
    label="Ridge Predicted Values",
    color="red",
    linestyle="--",
)
plt.title("Ridge Regression: True vs Predicted Values")
plt.xlabel("DATE")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()

# 绘制预测结果与真实值的对比图（Lasso 回归）
plt.figure(figsize=(12, 7))
plt.plot(y_test, label="True Values", color="blue")
plt.plot(
    y_pred_lasso,
    label="Lasso Predicted Values",
    color="green",
    linestyle="--",
)
plt.title("Lasso Regression: True vs Predicted Values")
plt.xlabel("DATE")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()
