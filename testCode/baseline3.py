import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense, LSTM

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"], index_col="Date")

# 2. 数据划分
train_data = data[:"2023-11-30"]
test_data = data["2023-12-01":"2023-12-31"]

# 3. 数据标准化
scaler = MinMaxScaler()
train_scaled = scaler.fit_transform(train_data)


# 4. 样本构造
def create_dataset(data, look_back=30, predict_forward=30):
    X, y = [], []
    for i in range(len(data) - look_back - predict_forward + 1):
        X.append(data[i : (i + look_back), 0])
        y.append(data[(i + look_back) : (i + look_back + predict_forward), 0])
    return np.array(X), np.array(y)


X_train, y_train = create_dataset(train_scaled)

# 5. 模型构建
model = Sequential()
model.add(LSTM(50, input_shape=(X_train.shape[1], 1)))
model.add(Dense(y_train.shape[1]))
model.compile(loss="mse", optimizer="adam")

# 6. 模型训练
model.fit(X_train, y_train, epochs=100, batch_size=32, verbose=2)

# 7. 预测
X_test, _ = create_dataset(scaler.transform(test_data.values))
y_pred_scaled = model.predict(X_test)

# 8. 逆向标准化
y_pred = scaler.inverse_transform(
    np.concatenate((y_pred_scaled, np.zeros((y_pred_scaled.shape[0], 1))), axis=1)
)[:, 0]

# 9. 打印值得关注的数据内容
print("预测的前5个数据点：", y_pred[:5])
print("真实值的前5个数据点：", test_data["FinalVolume"].values[:5])
