import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

# ===== Step 1: Load Data =====
# 假设你有一个CSV文件: fund_flow_simulated.csv，包含 [date, clus, in, out]
data = pd.read_csv("fund_flow_simulated.csv")
data["date"] = pd.to_datetime(data["date"])

# 按组别分组
groups = data.groupby("clus")

# Hyperparameters
window_size = 56  # 输入窗口长度
forecast_horizon = 14  # 预测未来14天
patch_size = 4
batch_size = 64
learning_rate = 0.001
epochs = 20

# ===== Step 2: Prepare Dataset (with Train-Test Split) =====
# 按日期排序
data = data.sort_values(by="date")

# 定义训练集和验证集的比例
train_ratio = 0.8
train_size = int(len(data) * train_ratio)

# 分割训练集和验证集
train_data = data[:train_size]
val_data = data[train_size:]
print(train_data.shape, val_data.shape)
print(f"Training set ends on: {train_data['date'].max()}")
print(f"Validation set starts on: {val_data['date'].min()}")

# 输出验证集的开始日期
print("Validation set starts on:", val_data["date"].min())


# 为训练集和验证集分别构建DataLoader
class TimeSeriesDataset(Dataset):
    def __init__(self, data, window_size, forecast_horizon):
        self.data = data
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon

    def __len__(self):
        return len(self.data) - self.window_size - self.forecast_horizon

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.window_size].values
        y = self.data[
            idx + self.window_size : idx + self.window_size + self.forecast_horizon
        ].values

        # 调试打印数据类型
        assert np.issubdtype(
            x.dtype, np.number
        ), f"Data contains non-numeric values: {x}"
        assert np.issubdtype(
            y.dtype, np.number
        ), f"Data contains non-numeric values: {y}"

        return torch.tensor(x, dtype=torch.float32), torch.tensor(
            y, dtype=torch.float32
        )


# 检查并转换数据类型
train_data[["in", "out"]] = train_data[["in", "out"]].apply(
    pd.to_numeric, errors="coerce"
)
val_data[["in", "out"]] = val_data[["in", "out"]].apply(pd.to_numeric, errors="coerce")

# 填充缺失值（根据业务需求选择合适的方法，例如用均值填充）
train_data = train_data.fillna(0)
val_data = val_data.fillna(0)

# 确保数据为数值类型
print(train_data[["in", "out"]].dtypes)
print(val_data[["in", "out"]].dtypes)

print(train_data[["in", "out"]].head())  # 查看前几行
print(train_data[["in", "out"]].dtypes)  # 检查数据类型

# 检查是否有缺失值
print(train_data.isnull().sum())
print(val_data.isnull().sum())


# 创建 DataLoader
train_dataset = TimeSeriesDataset(
    train_data[["in", "out"]],
    window_size=window_size,
    forecast_horizon=forecast_horizon,
)
val_dataset = TimeSeriesDataset(
    val_data[["in", "out"]], window_size=window_size, forecast_horizon=forecast_horizon
)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

for x_batch, y_batch in train_loader:
    print(f"x_batch shape: {x_batch.shape}")  # [batch_size, window_size, input_dim]
    print(
        f"y_batch shape: {y_batch.shape}"
    )  # [batch_size, forecast_horizon, output_dim]
    break


# ===== Step 3: Define PatchTST Model =====
class PatchTST(nn.Module):
    def __init__(
        self, input_dim, output_dim, window_size, forecast_horizon, patch_size
    ):
        super(PatchTST, self).__init__()
        self.patch_size = patch_size
        self.window_size = window_size
        self.forecast_horizon = forecast_horizon
        self.input_dim = input_dim
        print(self.patch_size, self.window_size, self.forecast_horizon, self.input_dim)
        assert (
            self.window_size % self.patch_size == 0
        ), "Window size must be divisible by patch size."

        # Patch Embedding: 将每个patch映射到128维的空间
        self.patch_embedding = nn.Linear(patch_size * input_dim, 128)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=128, nhead=4, batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=3)

        # Prediction Head: 最终预测的输出维度为 [batch_size, forecast_horizon, output_dim]
        self.fc = nn.Linear(128, output_dim)  # output_dim = 2, 即预测'in'和'out'

    def forward(self, x):
        print(
            f"Input to model: {x.shape}"
        )  # Expected: [batch_size, window_size, input_dim]

        B, T, D = x.shape  # B = batch_size, T = sequence_length, D = input_dim

        x = x.unfold(1, self.patch_size, self.patch_size).permute(
            0, 2, 1, 3
        )  # [batch_size, num_patches, patch_size, input_dim]
        # print(
        #     f"Unfolded shape: {x.shape}"
        # )  # Expected: [batch_size, patch_size, num_patches, input_dim]

        x = x.contiguous().view(B, -1, self.patch_size * D)
        # print(
        #     f"Flattened shape: {x.shape}"
        # )  # Expected: [B, num_patches, patch_size * input_dim]

        # Patch Embedding: 将每个patch映射到128维
        x = self.patch_embedding(x)  # [B, num_patches, 128]

        # Transformer Encoder
        x = self.transformer_encoder(x)  # [B, num_patches, 128]

        # Prediction: 通过全连接层进行映射
        x = self.fc(x)  # [B, num_patches, 2]

        # 输出维度 [B, forecast_horizon, 2]，即预测14个时间步的'in'和'out'
        # 假设每个patch的时间步为forecast_horizon个时间步
        # 通过对num_patches进行重塑使输出与预测期望一致
        x = x[:, : self.forecast_horizon, :]  # 只取前14个时间步 [B, 14, 2]

        return x


# ===== Step 4: Training and Evaluation =====
model = PatchTST(
    input_dim=2,
    output_dim=2,
    window_size=window_size,
    forecast_horizon=forecast_horizon,
    patch_size=patch_size,
)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# 使用学习率调度器
scheduler = ReduceLROnPlateau(optimizer, "min", patience=3, verbose=True)

# Training Loop
for epoch in range(epochs):
    model.train()
    epoch_loss = 0.0
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        predictions = model(x_batch)
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()

    # 验证集评估
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for x_batch, y_batch in val_loader:
            predictions = model(x_batch)
            loss = criterion(predictions, y_batch)
            val_loss += loss.item()

    scheduler.step(val_loss)  # 调整学习率
    print(
        f"Epoch {epoch + 1}, Train Loss: {epoch_loss:.4f}, Validation Loss: {val_loss:.4f}"
    )

# ===== Step 5: Prediction =====
# 获取验证集上的预测结果
model.eval()
val_predictions = []
val_actuals = []
with torch.no_grad():
    for x_batch, y_batch in val_loader:
        pred = model(x_batch)
        val_predictions.append(pred.numpy())
        val_actuals.append(y_batch.numpy())

# 将预测结果转换成DataFrame，并与真实结果对比
val_predictions = np.concatenate(val_predictions, axis=0)  # (77, 14, 2)
val_actuals = np.concatenate(val_actuals, axis=0)  # (77, 14, 2)

# 方法 1: 展平时间步
val_predictions = val_predictions.reshape(val_predictions.shape[0], -1)
val_df = pd.DataFrame(
    val_predictions,
    columns=[f"{col}_{i}" for i in range(14) for col in ["in", "out"]],
)

# 或者使用方法 2: 长表格形式
# time_steps = val_predictions.shape[1]
# val_df = pd.DataFrame({
#     "step": np.tile(np.arange(time_steps), val_predictions.shape[0]),
#     "in": val_predictions[:, :, 0].flatten(),
#     "out": val_predictions[:, :, 1].flatten(),
#     "actual_in": np.repeat(val_actuals[:, :, 0], time_steps),
#     "actual_out": np.repeat(val_actuals[:, :, 1], time_steps),
# })

print(val_df)


# ===== Step 6: Make Future Predictions =====
# 假设你已经训练完模型，进行未来的14天预测
model.eval()
future_predictions = []
group_keys = []  # 用于存储分组键

for clus, group in groups:
    group_keys.append(clus)  # 存储分组键

    # 修复：仅选择数值列传递给 TimeSeriesDataset
    group_data = group[["in", "out"]]  # 去掉 'date' 列，只保留数值列
    group_dataset = TimeSeriesDataset(
        group_data, window_size=window_size, forecast_horizon=forecast_horizon
    )
    group_loader = DataLoader(group_dataset, batch_size=1, shuffle=False)

    # 遍历数据加载器，进行预测
    group_preds = []
    for x_batch, _ in group_loader:
        with torch.no_grad():
            pred = model(x_batch)
            group_preds.append(pred.numpy())

    # 将每组预测结果保存为单独的列表
    future_predictions.append(group_preds)

# 展平预测结果，同时保留分组信息和时间步
group_ids = []  # 用于存储分组键
dates = []       # 用于存储预测的日期
flat_predictions = []  # 用于存储展平后的预测值

# 遍历每组的预测结果
for clus, preds in zip(group_keys, future_predictions):  # group_keys: 分组键，future_predictions: shape (num_groups, ?)
    group_data = groups.get_group(clus)  # 获取当前分组的数据
    last_date = group_data["date"].max()  # 当前分组的最后日期
    pred_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=len(preds),  # 确保生成的日期数和预测结果一致
        freq="D",
    )

    # 展平预测结果
    for t, pred in enumerate(preds):
        group_ids.append(clus)  # 添加分组键
        dates.append(pred_dates[t])  # 添加对应的预测日期
        flat_predictions.append(pred)  # 添加预测值

# 展平预测结果
flat_predictions = np.array(flat_predictions)  # 确保是 numpy 数组

# ===== 方法 1：展平时间步 =====
# 将 (661, 14, 2) 转换为 (661*14, 2)
flat_predictions = flat_predictions.reshape(-1, 2)  # 将嵌套的列表转换为二维数组

# 创建时间步索引
time_steps = np.tile(np.arange(1, 15), len(flat_predictions) // 14)  # 每组14个时间步
group_ids = np.repeat(group_ids, 14)  # 每个分组重复14次
dates = np.repeat(dates, 14)  # 每个预测日期重复14次

# 创建 DataFrame
future_df = pd.DataFrame(
    flat_predictions,
    columns=["in", "out"],
)
future_df["group_id"] = group_ids
future_df["date"] = dates
future_df["time_step"] = time_steps

print(future_df)

# ===== 方法 2（如果需要宽格式表格）=====
# 如果需要宽格式表格，取消注释以下代码并注释掉方法 1
# 将 (661, 14, 2) 转换为 (661, 28)
# flat_predictions = flat_predictions.reshape(flat_predictions.shape[0], -1)  # 转换为宽格式
# 
# # 生成列名
# columns = [f"{metric}_t{t+1}" for t in range(14) for metric in ["in", "out"]]
# 
# # 创建 DataFrame
# future_df = pd.DataFrame(flat_predictions, columns=columns)
# future_df["group_id"] = group_ids
# future_df["date"] = dates
# 
# print(future_df)


import matplotlib.pyplot as plt

# 获取第一个分组的14天完整预测数据
group_id_to_plot = "clus"
group_data = future_df[future_df["group_id"] == group_id_to_plot]

# 按日期分组并取出前14天数据
first_14_days = group_data["date"].unique()[:14]
plot_data = group_data[group_data["date"].isin(first_14_days)]

# 可视化
plt.figure(figsize=(14, 8))

# 分别绘制 in 和 out 的预测值
for var in ["in", "out"]:
    for date in first_14_days:
        daily_data = plot_data[plot_data["date"] == date]
        plt.plot(daily_data["time_step"], daily_data[var], label=f"{date} - {var}")

# 图表设置
plt.title(f"14-Day Prediction for Group {group_id_to_plot}", fontsize=16)
plt.xlabel("Time Step", fontsize=14)
plt.ylabel("Predicted Values", fontsize=14)
plt.legend(loc="upper left", bbox_to_anchor=(1.05, 1), fontsize=10)
plt.grid(True)
plt.tight_layout()
plt.show()
