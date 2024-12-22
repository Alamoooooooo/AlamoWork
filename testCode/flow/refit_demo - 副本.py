import cudf
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
import pickle
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

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
# Define features and target
features = ["lag_1", "lag_2", "rolling_mean_3", "rolling_std_3"]
target = "net"

# Split data into train and test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=False)
train_X, train_y = train_df[features], train_df[target]
test_X, test_y = test_df[features], test_df[target]

# Create dataset for LightGBM
lgb_train = lgb.Dataset(train_X, train_y, device="gpu")
lgb_valid = lgb.Dataset(test_X, test_y, reference=lgb_train, device="gpu")

# Define parameters for LightGBM
params = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "verbose": -1,
    "device": "gpu",  # Ensure training uses GPU
}

# Use early stopping and log evaluation during training
from lightgbm import early_stopping, log_evaluation

callbacks = [
    early_stopping(stopping_rounds=50),  # Stop if no improvement in 50 rounds
    log_evaluation(period=50),  # Log evaluation every 50 rounds
]

# Train the model
model = lgb.train(
    params,
    lgb_train,
    valid_sets=[lgb_train, lgb_valid],
    num_boost_round=1000,
    callbacks=callbacks,
)

# Step 4: Refit with new online data (GPU-enabled)
online_data = test_df.sample(frac=0.2, random_state=42)
online_X, online_y = online_data[features], online_data[target]

# Ensure feature consistency between training and new data
assert list(online_X.columns) == list(train_X.columns), "Feature mismatch!"

# Validate that there are no missing values
if online_X.isnull().any().any() or online_y.isnull().any():
    raise ValueError("New data contains missing values.")

# Create dataset for new online data
lgb_online = lgb.Dataset(online_X, label=online_y, device="gpu")

# Backup the model before refitting
with open("model_backup.pkl", "wb") as f:
    pickle.dump(model, f)

# Update the model with new data using refit (on GPU)
model = model.refit(online_X, online_y, reuse_weights=True, device="gpu")

# Step 5: Prediction on new data
new_data = test_df.sample(frac=0.1, random_state=24)
new_X = new_data[features]

# Predict using the refitted model
predictions = model.predict(new_X)

# Evaluate the model's performance
mse = mean_squared_error(new_data[target], predictions)
print(f"Updated Model MSE: {mse}")
