import gc
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from datetime import timedelta
from sklearn.metrics import mean_squared_error, mean_absolute_error

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")

# 2. 数据划分
PRED_STEP_LEN = 60
sd = "2023-09-15"
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sd:ed]


# 3. 样本构造函数
def prepare_sample_time(df_bal_class, tm_col="DATE", shift_steps=range(1, 33)):
    df_bal_class = df_bal_class.reset_index(drop=False)[["Date", "FinalVolume"]]
    df_bal_class.columns = ["DATE", "bal"]
    df_bal_class_new = []
    df_bal_class = df_bal_class.sort_values([tm_col], ascending=True).reset_index(
        drop=True
    )
    df_dt = df_bal_class[[tm_col]].drop_duplicates()

    # 造steps步长样本
    for i in shift_steps:
        df_dt[f"{tm_col}_target"] = df_dt[tm_col] + timedelta(days=i)
        df_dt["shift_steps"] = i
        df_bal_class = df_bal_class.merge(df_dt, how="left", on=[tm_col])
        df_bal_class = df_bal_class.merge(
            df_bal_class[[tm_col, "bal"]].rename(
                columns={tm_col: f"{tm_col}_target", "bal": "bal_target"}
            ),
            how="left",
            on=[f"{tm_col}_target"],
        )
        df_bal_class["bal_target_diff"] = (
            df_bal_class["bal_target"] - df_bal_class["bal"]
        )
        df_bal_class_new.append(df_bal_class.copy())
        df_bal_class.drop(
            [f"{tm_col}_target", "bal_target_diff", "bal_target", "shift_steps"],
            axis=1,
            inplace=True,
        )
    del df_bal_class
    gc.collect()

    df_bal_class_new = pd.concat(df_bal_class_new, axis=0, ignore_index=True)
    df_bal_class = df_bal_class_new.sort_values(
        [tm_col, "shift_steps"], ascending=True
    ).reset_index(drop=True)
    return df_bal_class


trian_df = prepare_sample_time(train_data)
test_df = prepare_sample_time(test_data)

print(trian_df)
print(test_df)

step = PRED_STEP_LEN


def feature_gen(input_df):
    df = input_df.copy()

    # 特征工程：生成滞后特征和日期特征
    df["Year"] = df["DATE"].dt.year
    df["Month"] = df["DATE"].dt.month
    df["Day"] = df["DATE"].dt.day
    # 添加工作日和周末标记
    df["DayOfWeek"] = df["DATE"].dt.dayofweek
    df["IsWeekend"] = np.where(df["DayOfWeek"] >= 5, 1, 0)
    # 生成滞后特征
    # （假设我们使用过去lag天的数据）
    for i in range(1, step):
        df.loc[:, f"Lag_{i}"] = df["bal"].shift(i)

    # 生成滚动平均特征（7天窗口）
    df.loc[:, "Rolling_Mean"] = df["bal"].rolling(window=7).mean()
    return df


train_df = feature_gen(trian_df)
test_df = feature_gen(test_df)
# 填充因滚动计算而产生的NA值
train_df.fillna(-999, inplace=True)
test_df.fillna(-999, inplace=True)
# train_df = train_df.dropna()
# test_df = test_df.dropna()
train_df = train_df.query("DATE_target<=@sd").sort_values("DATE_target")
test_df = test_df.query("DATE_target<=@ed").sort_values("DATE_target")
nd = (pd.to_datetime(sd) + timedelta(days=1)).strftime("%Y-%m-%d")
test_df = test_df.query("DATE==@sd")


def create_dataset(data, feature_cols, target_col):
    X = data[feature_cols]
    y = data[target_col]
    return np.array(X), np.array(y)


# 4. 构造训练集和测试集
target_col = "bal_target"
feature_cols = [
    "bal",
    # "shift_steps",
    # "bal_target_diff",
    "Year",
    "Month",
    "Day",
    "DayOfWeek",
    "IsWeekend",
    # "Rolling_Mean",
] + [f"Lag_{i}" for i in range(1, step)]
X_train, y_train = create_dataset(train_df, feature_cols, target_col)
X_test, y_test = create_dataset(test_df, feature_cols, target_col)

# 打印检查 X_test 的内容
print("X_test shape:", X_test.shape)
print("X_test:", X_test)

# 5. 确保输入数据是二维的
if X_test.size > 0:
    X_test = X_test.reshape(-1, X_test.shape[1])
else:
    raise ValueError("X_test is empty. Please check your dataset or parameters.")

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
test_df_sorted = test_df.sort_values("DATE_target")
print(test_df_sorted)

# 绘制预测结果与真实值的对比图
plt.figure(figsize=(14, 7))
plt.plot(
    test_df_sorted["DATE_target"],
    test_df_sorted["bal_target"],
    label="True Values",
    color="blue",
)
plt.plot(
    test_df_sorted["DATE_target"],
    test_df_sorted["Predicted"],
    label="Predicted Values",
    color="red",
    linestyle="--",
)
plt.title("True vs Predicted Values")
plt.xlabel("DATE_target")
plt.ylabel("Volume")
plt.legend()
plt.grid(True)
plt.show()
