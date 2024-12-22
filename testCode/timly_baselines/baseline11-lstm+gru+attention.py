import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, GRU
from tensorflow.keras.optimizers import Adam
from datetime import timedelta
from tensorflow.keras.layers import Layer
import tensorflow.keras.backend as K

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 确保数据按照日期排序
data.set_index("DATE", inplace=True)

# 2. 数据划分
PRED_STEP_LEN = 60
sd = "2023-10-15"
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


time_steps = 7
X_train, y_train = create_sequences(scaled_train_data, time_steps)
X_test, y_test = create_sequences(scaled_test_data, time_steps)

# Reshape for LSTM/GRU (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))


# 修正后的 Attention 层
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        # input_shape[-1] 是输入的特征维度
        self.W = self.add_weight(
            shape=(input_shape[-1], input_shape[-1]),
            initializer="glorot_uniform",
            trainable=True,
        )
        self.b = self.add_weight(
            shape=(input_shape[-1],), initializer="zeros", trainable=True
        )
        self.u = self.add_weight(
            shape=(input_shape[-1], 1), initializer="glorot_uniform", trainable=True
        )
        super(Attention, self).build(input_shape)

    def call(self, x):
        # 计算注意力得分
        uit = K.tanh(K.dot(x, self.W) + self.b)  # [batch_size, time_steps, features]
        ait = K.dot(uit, self.u)  # [batch_size, time_steps, 1]
        ait = K.squeeze(ait, -1)  # 将最后一个维度降维，变为 [batch_size, time_steps]
        ait = K.softmax(ait, axis=1)  # 归一化为权重 [batch_size, time_steps]
        ait = K.expand_dims(ait)  # [batch_size, time_steps, 1]
        weighted_input = x * ait  # 加权输入 [batch_size, time_steps, features]
        return K.sum(weighted_input, axis=1)  # 输出 [batch_size, features]

    def compute_output_shape(self, input_shape):
        # 输出形状为 [batch_size, features]
        return (input_shape[0], input_shape[-1])


# 重新定义LSTM模型，加入Attention层
def build_lstm_attention_model(time_steps):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(LSTM(64, return_sequences=True))
    model.add(Attention())  # 加入自定义Attention层
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


# 重新定义GRU模型，加入Attention层
def build_gru_attention_model(time_steps):
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=(time_steps, 1)))
    model.add(GRU(64, return_sequences=True))
    model.add(Attention())  # 加入自定义Attention层
    model.add(Dense(1))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


# 训练和评估LSTM + Attention模型
lstm_attention_model = build_lstm_attention_model(time_steps)
lstm_attention_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

lstm_attention_predictions = lstm_attention_model.predict(X_test)
lstm_attention_predictions = scaler.inverse_transform(lstm_attention_predictions)

# 训练和评估GRU + Attention模型
gru_attention_model = build_gru_attention_model(time_steps)
gru_attention_model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

gru_attention_predictions = gru_attention_model.predict(X_test)
gru_attention_predictions = scaler.inverse_transform(gru_attention_predictions)

# 可视化带有Attention机制的预测结果
plt.figure(figsize=(12, 6))
plt.plot(
    test_data.index[time_steps:],
    test_data["bal"][time_steps:],
    label="True Values",
    color="blue",
)
plt.plot(
    test_data.index[time_steps:],
    lstm_attention_predictions,
    label="LSTM + Attention Predictions",
    color="red",
    linestyle="--",
)
plt.plot(
    test_data.index[time_steps:],
    gru_attention_predictions,
    label="GRU + Attention Predictions",
    color="green",
    linestyle="--",
)
plt.title("True Values vs LSTM/GRU with Attention Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()
