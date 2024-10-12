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
from tensorflow.keras.layers import Dropout

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 确保数据按照日期排序
data.set_index("DATE", inplace=True)

# 调整后的参数设置
PRED_STEP_LEN = 130  # 增加测试集长度，确保足够的多步预测窗口
PREDICTION_LENGTH = 45  # 预测未来的步数
time_steps = 7  # 增加输入的时间步长度，让模型参考更多的历史数据

# 2. 数据划分
sd = "2023-07-15"
sdd = (pd.to_datetime(sd) + timedelta(days=1)).strftime("%Y-%m-%d")
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sdd:ed]

# 3. 数据标准化
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_train_data = scaler.fit_transform(train_data[["bal"]])
scaled_test_data = scaler.transform(test_data[["bal"]])


# 4. 生成 LSTM 和 GRU 所需的序列数据（多步预测）
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

# Reshape for LSTM/GRU (samples, time_steps, features)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# 打印数据形状
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")


# 5. Attention层定义
class Attention(Layer):
    def __init__(self, **kwargs):
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
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
        uit = K.tanh(K.dot(x, self.W) + self.b)  # [batch_size, time_steps, features]
        ait = K.dot(uit, self.u)  # [batch_size, time_steps, 1]
        ait = K.squeeze(ait, -1)  # [batch_size, time_steps]
        ait = K.softmax(ait, axis=1)  # [batch_size, time_steps]
        ait = K.expand_dims(ait)  # [batch_size, time_steps, 1]
        weighted_input = x * ait  # 加权输入 [batch_size, time_steps, features]
        return K.sum(weighted_input, axis=1)  # 输出 [batch_size, features]

    def compute_output_shape(self, input_shape):
        return (input_shape[0], input_shape[-1])


# 6. 构建LSTM模型，加入Attention层，支持多步预测
def build_lstm_attention_model(time_steps, prediction_length):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape=(time_steps, 1)))
    # model.add(Dropout(0.2))  # 添加 Dropout 层
    model.add(LSTM(64, return_sequences=True))
    # model.add(Dropout(0.2))  # 添加 Dropout 层
    model.add(Attention())
    model.add(Dense(prediction_length))
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


# 7. 训练和评估LSTM + Attention模型
lstm_attention_model = build_lstm_attention_model(time_steps, PREDICTION_LENGTH)
lstm_attention_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

# 使用模型进行多步预测
lstm_attention_predictions = lstm_attention_model.predict(X_test)
# 反标准化预测结果
lstm_attention_predictions = scaler.inverse_transform(lstm_attention_predictions)


# 8. 训练和评估GRU + Attention模型
def build_gru_attention_model(time_steps, prediction_length):
    model = Sequential()
    model.add(GRU(128, return_sequences=True, input_shape=(time_steps, 1)))  # 保持三维
    # model.add(Dropout(0.2))  # 添加 Dropout 层
    model.add(GRU(64, return_sequences=True))  # 保持三维以传递给 Attention 层
    # model.add(Dropout(0.2))  # 添加 Dropout 层
    model.add(Attention())  # 加入 Attention 层
    model.add(Dense(prediction_length))  # 输出多个时间步
    model.compile(optimizer=Adam(learning_rate=0.001), loss="mean_squared_error")
    return model


gru_attention_model = build_gru_attention_model(time_steps, PREDICTION_LENGTH)
gru_attention_model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1,
)

gru_attention_predictions = gru_attention_model.predict(X_test)
gru_attention_predictions = scaler.inverse_transform(gru_attention_predictions)

# 9. 可视化LSTM + Attention的多步预测结果
plt.figure(figsize=(12, 6))

# 多步预测的每个时间步的预测值需要正确对齐
for i in range(PREDICTION_LENGTH):
    plt.plot(
        test_data.index[
            time_steps + i : len(lstm_attention_predictions) + time_steps + i
        ],
        lstm_attention_predictions[:, i],
        label=f"LSTM + Attention Prediction (step {i+1})",
        linestyle="--",
    )

plt.plot(
    test_data.index[time_steps:],
    test_data["bal"][time_steps:],
    label="True Values",
    color="blue",
)

plt.title("True Values vs LSTM with Attention Multi-Step Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


# 10. 可视化GRU + Attention的多步预测结果
plt.figure(figsize=(12, 6))

# 多步预测的每个时间步的预测值需要正确对齐
for i in range(PREDICTION_LENGTH):
    plt.plot(
        test_data.index[
            time_steps + i : len(lstm_attention_predictions) + time_steps + i
        ],
        lstm_attention_predictions[:, i],
        label=f"LSTM + Attention Prediction (step {i+1})",
        linestyle="--",
    )

plt.plot(
    test_data.index[time_steps:],
    test_data["bal"][time_steps:],
    label="True Values",
    color="blue",
)

plt.title("True Values vs LSTM with Attention Multi-Step Predictions")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# # 定义权重（越接近当前时间步的权重越小）
# weights = np.array([0.1, 0.3, 0.6])

# # 对前 3 个时间步的预测值进行加权平均
# weighted_first_3_predictions = np.average(
#     lstm_attention_predictions[:, :3], axis=1, weights=weights
# )

# # 创建时间步对齐的索引数组
# aligned_indices = test_data.index[
#     time_steps : len(lstm_attention_predictions) + time_steps
# ]

# # 可视化
# plt.figure(figsize=(12, 6))
# plt.plot(
#     test_data.index[time_steps:],
#     test_data["bal"][time_steps:],
#     label="True Values",
#     color="blue",
# )

# # 在时间步上对齐预测结果
# plt.plot(
#     aligned_indices,  # 正确对齐时间步
#     weighted_first_3_predictions,  # 加权平均的预测值
#     label="LSTM + Attention Weighted First 3 Steps Prediction",
#     linestyle="--",
#     color="green",
# )

# plt.title("True Values vs LSTM with Attention Weighted First 3 Steps Prediction")
# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()

# 只绘制第一个时间步的预测
plt.figure(figsize=(12, 6))
plt.plot(
    test_data.index[time_steps:],
    test_data["bal"][time_steps:],
    label="True Values",
    color="blue",
)
plt.plot(
    test_data.index[time_steps : len(lstm_attention_predictions) + time_steps],
    lstm_attention_predictions[:, 0],  # 选择第一个时间步的预测值
    label="LSTM + Attention Final Step Prediction",
    linestyle="--",
    color="red",
)
plt.title("True Values vs LSTM with Attention Final Step Prediction")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()


# # 使用所有时间步预测的平均值
# average_predictions = np.mean(lstm_attention_predictions, axis=1)

# plt.figure(figsize=(12, 6))
# plt.plot(
#     test_data.index[time_steps:],
#     test_data["bal"][time_steps:],
#     label="True Values",
#     color="blue",
# )
# plt.plot(
#     test_data.index[time_steps : len(lstm_attention_predictions) + time_steps],
#     average_predictions,  # 使用所有时间步预测的平均值
#     label="LSTM + Attention Average Prediction",
#     linestyle="--",
#     color="red",
# )
# plt.title("True Values vs LSTM with Attention Average Prediction")
# plt.xlabel("Date")
# plt.ylabel("Value")
# plt.legend()
# plt.grid(True)
# plt.show()


# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from matplotlib.animation import FuncAnimation

# # 假设你已经有了lstm_attention_predictions 和 test_data等数据

# # 设置动画
# fig, ax = plt.subplots(figsize=(12, 6))


# # 初始化图表函数
# def init2():
#     ax.clear()
#     ax.plot(
#         range(len(test_data["bal"][time_steps:])),
#         test_data["bal"][time_steps:],
#         label="True Values",
#         color="blue",
#     )
#     ax.set_title("True Values vs LSTM Multi-Step Predictions")
#     ax.set_xlabel("Time Step")
#     ax.set_ylabel("Value")
#     ax.grid(True)
#     ax.legend()


# # 更新函数，用于逐帧绘制每个时间步的预测值
# def update2(step):
#     ax.clear()  # 每次绘制前清空当前图像
#     init2()  # 初始化绘制真实值

#     # 动态展示逐步增加的预测结果
#     for i in range(step + 1):  # 每次增加一个时间步的预测
#         ax.plot(
#             range(
#                 len(lstm_attention_predictions)
#             ),  # 不再使用对齐时间步，直接绘制预测值
#             lstm_attention_predictions[:, i],  # 当前时间步的预测值
#             linestyle="--",
#             label=f"Prediction (step {i+1})",
#             color=f"C{i % 10}",  # 使用不同的颜色绘制每个时间步的预测值
#         )
#     ax.legend()


# # 创建动画
# ani = FuncAnimation(
#     fig, update2, frames=PREDICTION_LENGTH, init_func=init2, blit=False, repeat=False
# )

# # 将动画保存为 GIF
# ani.save("lstm_multi_step_prediction_no_alignment.gif", writer="imagemagick")

# # 显示动画
# plt.show()


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# 假设你已经有了 lstm_attention_predictions 和 test_data 等数据

# 创建时间步对齐的索引数组
aligned_indices = test_data.index[
    time_steps : len(lstm_attention_predictions) + time_steps
]

# 设置动画
fig, ax = plt.subplots(figsize=(12, 6))


# 初始化图表函数
def init():
    ax.clear()
    ax.plot(
        test_data.index[time_steps:],
        test_data["bal"][time_steps:],
        label="True Values",
        color="blue",
    )
    ax.set_title("True Values vs LSTM Multi-Step Predictions (Single Line)")
    ax.set_xlabel("Date")  # 使用日期作为横轴标签
    ax.set_ylabel("Value")
    ax.grid(True)
    ax.legend()


# 更新函数：每帧只显示一个时间步的预测结果
def update(step):
    ax.clear()  # 清空当前图像
    init()  # 重新绘制真实值

    # 只绘制当前时间步的预测结果，不叠加之前的
    ax.plot(
        aligned_indices,  # 恢复时间步对齐
        lstm_attention_predictions[:, step],  # 当前时间步的预测值
        linestyle="--",
        label=f"Prediction (step {step+1})",
        color=f"C{step % 10}",  # 使用不同的颜色绘制每个时间步的预测值
    )
    ax.legend()


# 创建动画，增加 interval 参数来减慢速度
ani = FuncAnimation(
    fig,
    update,
    frames=PREDICTION_LENGTH,
    init_func=init,
    blit=False,
    repeat=False,
    interval=1000,  # 每帧间隔 1000 毫秒 (1 秒)
)

# 将动画保存为 GIF
ani.save("lstm_single_line_prediction_slow_with_alignment.gif", writer="imagemagick")

# 显示动画
plt.show()
