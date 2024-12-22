import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Step 1: Simulate data generation
np.random.seed(42)
dates = pd.date_range(start="2020-01-01", end="2023-01-01", freq="D")
gids = ["A", "B", "C"]
keys = ["Group1", "Group2", "Group3"]

data = []
for gid in gids:
    for key in keys:
        in_series = np.random.poisson(lam=20, size=len(dates))
        out_series = np.random.poisson(lam=15, size=len(dates))
        data.append(
            pd.DataFrame(
                {
                    "date": dates,
                    "gid": gid,
                    "key": key,
                    "in": in_series,
                    "out": out_series,
                }
            )
        )

df = pd.concat(data).reset_index(drop=True)

# Step 2: Feature engineering
df["net"] = df["in"] - df["out"]
df = df.sort_values(by=["gid", "key", "date"])
df["lag_1"] = df.groupby(["gid", "key"])["net"].shift(1)
df["lag_2"] = df.groupby(["gid", "key"])["net"].shift(2)
df["rolling_mean_3"] = df.groupby(["gid", "key"])["net"].transform(
    lambda x: x.rolling(window=3).mean()
)
df["rolling_std_3"] = df.groupby(["gid", "key"])["net"].transform(
    lambda x: x.rolling(window=3).std()
)

# Remove NaN rows caused by lagging and rolling features
df.dropna(inplace=True)

# Step 3: Train LightGBM model
features = ["lag_1", "lag_2", "rolling_mean_3", "rolling_std_3"]
target = "net"

train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
train_X, train_y = train_df[features], train_df[target]
test_X, test_y = test_df[features], test_df[target]

# 定义训练集和验证集
lgb_train = lgb.Dataset(train_X, train_y)
lgb_valid = lgb.Dataset(test_X, test_y, reference=lgb_train)

params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": -1,
}

from lightgbm import early_stopping, log_evaluation

# 使用回调函数实现早停和日志
callbacks = [
    early_stopping(stopping_rounds=50),  # 50轮内未改善则停止训练
    log_evaluation(period=50),  # 每50轮输出一次日志
]

# 训练模型
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],  # 验证集指定
    num_boost_round=1000,  # 最大迭代次数
    callbacks=callbacks,  # 使用回调函数
)

# Step 4: Refit with new online data
online_data = test_df.sample(frac=0.2, random_state=42)
online_X, online_y = online_data[features], online_data[target]

# 确保特征顺序一致
assert list(online_X.columns) == list(train_X.columns), "Feature mismatch!"
# 模型更新前验证数据质量： 确保新数据无缺失值或异常值
if online_X.isnull().any().any() or online_y.isnull().any():
    raise ValueError("New data contains missing values.")


# 创建包含新数据的 Dataset
lgb_online = lgb.Dataset(online_X, label=online_y)

# 回滚策略： 在更新前备份模型
import pickle

with open("model_backup.pkl", "wb") as f:
    pickle.dump(model, f)


# 调用 refit 方法更新模型
model = model.refit(online_X, online_y, reuse_weights=True)


# Step 5: Prediction
new_data = test_df.sample(frac=0.1, random_state=24)
new_X = new_data[features]
predictions = model.predict(new_X)

# Return metrics and predictions
mse = mean_squared_error(new_data[target], predictions)
# mse, predictions[:5]
print(f"Updated Model MSE: {mse}")
