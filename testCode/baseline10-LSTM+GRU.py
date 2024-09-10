import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
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

# 3. 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[["bal"]])
scaled_test_data = scaler.transform(test_data[["bal"]])


# 4. 生成 LSTM 和 GRU 所需的序列数据
def create_sequences(data, time_steps):
    X, y = [], []
    for i in range(len(data) - time_steps):
        X.append(data[i : (i + time_steps), 0])
        y.append(data[i + time_steps, 0])
    return np.array(X), np.array(y)


time_steps = 30
X_train, y_train = create_sequences(scaled_train_data, time_steps)
X_test, y_test = create_sequences(scaled_test_data, time_steps)

# Reshape for LSTM/GRU (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Print the shapes
print("===X_train shape and y_train shape:")
print(X_train.shape, y_train.shape)
print("===X_test shape and y_test shape:")
print(X_test.shape, y_test.shape)


# 5. 构建LSTM模型
def build_lstm_model():
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(64))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_absolute_error")
    return model


# 6. 构建GRU模型
def build_gru_model():
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(GRU(64))
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_absolute_error")
    return model


# 7. 训练和预测LSTM模型
lstm_model = build_lstm_model()
lstm_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

lstm_predictions = lstm_model.predict(X_test)
lstm_predictions = scaler.inverse_transform(lstm_predictions)  # 反标准化

# 8. 训练和预测GRU模型
gru_model = build_gru_model()
gru_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

gru_predictions = gru_model.predict(X_test)
gru_predictions = scaler.inverse_transform(gru_predictions)  # 反标准化


# 9. 评估模型性能
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    print(f"{model_name} - MSE: {mse:.2f}, RMSE: {rmse:.2f}, MAE: {mae:.2f}")


print("===lstm_predictions:")
print(lstm_predictions)
print("===test_data[time_steps:]:")
print(test_data[time_steps:])
print("y_test:")
print(y_test)


evaluate_model(test_data[time_steps:], lstm_predictions, "LSTM")
evaluate_model(test_data[time_steps:], gru_predictions, "GRU")

# 10. 可视化结果
plt.figure(figsize=(12, 6))
plt.plot(
    test_data.index[time_steps:],
    test_data["bal"][time_steps:],
    label="True Values",
    color="blue",
)
plt.plot(
    test_data.index[time_steps:],
    lstm_predictions,
    label="LSTM Predictions",
    color="red",
    linestyle="--",
)
plt.plot(
    test_data.index[time_steps:],
    gru_predictions,
    label="GRU Predictions",
    color="green",
    linestyle="--",
)
plt.title("True Values vs LSTM/GRU Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
