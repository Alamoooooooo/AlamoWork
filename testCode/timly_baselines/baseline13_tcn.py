import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from tensorflow_addons.layers import TCN  # TCN layer from tensorflow-addons
from datetime import timedelta

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 确保数据按照日期排序
data.set_index("DATE", inplace=True)

# 2. 参数设置
PRED_STEP_LEN = 130  # 测试集长度
PREDICTION_LENGTH = 45  # 预测未来的步数
time_steps = 7  # 输入的时间步长度

# 3. 数据划分
sd = "2023-05-15"
sdd = (pd.to_datetime(sd) + timedelta(days=1)).strftime("%Y-%m-%d")
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sdd:ed]

# 4. 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[["bal"]])
scaled_test_data = scaler.transform(test_data[["bal"]])


# 5. 生成 TCN 所需的序列数据（多步预测）
def create_sequences_multi_step(data, time_steps, prediction_length):
    X, y = [], []
    for i in range(len(data) - time_steps - prediction_length):
        X.append(data[i : (i + time_steps), 0])
        y.append(data[(i + time_steps) : (i + time_steps + prediction_length), 0])
    return np.array(X), np.array(y)


# 生成训练和测试集
X_train, y_train = create_sequences_multi_step(
    scaled_train_data, time_steps, PREDICTION_LENGTH
)
X_test, y_test = create_sequences_multi_step(
    scaled_test_data, time_steps, PREDICTION_LENGTH
)

# Reshape for TCN (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 打印数据形状
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# 6. 构建TCN模型
def build_tcn_model(time_steps, prediction_length):
    model = Sequential()
    model.add(
        TCN(input_shape=(time_steps, 1), dilations=[1, 2, 4, 8])
    )  # TCN层，带有膨胀卷积
    model.add(Dense(prediction_length))  # 输出层，预测未来多个时间步
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


# 训练和评估TCN模型
tcn_model = build_tcn_model(time_steps, PREDICTION_LENGTH)
tcn_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# 使用模型进行多步预测
tcn_predictions = tcn_model.predict(X_test)
# 反标准化预测结果
tcn_predictions = scaler.inverse_transform(tcn_predictions)

# 7. 可视化TCN的多步预测结果
plt.figure(figsize=(12, 6))

# 多步预测的每个时间步的预测值需要正确对齐
for i in range(PREDICTION_LENGTH):
    plt.plot(
        test_data.index[time_steps + i : len(tcn_predictions) + time_steps + i],
        tcn_predictions[:, i],
        label=f"TCN Prediction (step {i+1})",
        linestyle="--",
    )

plt.plot(
    test_data.index[time_steps:],
    test_data["bal"][time_steps:],
    label="True Values",
    color="blue",
)

plt.title("True Values vs TCN Multi-Step Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
