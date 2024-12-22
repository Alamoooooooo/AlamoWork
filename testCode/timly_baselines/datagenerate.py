import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 设定随机种子以确保结果可重复
np.random.seed(42)

# 生成日期范围，假设是2022-2023年全年的数据
date_range = pd.date_range(start="2022-01-01", end="2023-12-31", freq="D")

# 创建空的 DataFrame 来存储数据
df = pd.DataFrame({"Date": date_range})

# 添加工作日和周末标记
df["DayOfWeek"] = df["Date"].dt.dayofweek
df["IsWeekend"] = df["DayOfWeek"] >= 5

# 生成基本交易量数据
base_volume = 1000
df["BaseVolume"] = base_volume + np.random.normal(0, 200, len(df))  # 添加一定的随机性

# 调整工作日和周末的交易量
df["AdjustedVolume"] = df["BaseVolume"] * np.where(df["IsWeekend"], 0.5, 1.5)

# 定义一些节假日
holidays = [
    "2023-01-01",  # 元旦
    "2023-05-01",  # 劳动节
    "2023-10-01",  # 国庆节
    "2023-12-25",  # 圣诞节
    # 其他节假日可以继续添加
]

df["IsHoliday"] = df["Date"].isin(pd.to_datetime(holidays))

# 在节假日及其前后的交易量调整
for offset in range(-2, 3):  # 节假日当天及前后两天
    df.loc[
        df["Date"].isin(pd.to_datetime(holidays) + pd.Timedelta(days=offset)),
        "AdjustedVolume",
    ] *= 1.8

# 添加季节性因素
df["Month"] = df["Date"].dt.month
df["SeasonalAdjustment"] = 1 + 0.1 * np.sin(
    2 * np.pi * df["Month"] / 12
)  # 简单的季节性调整
df["FinalVolume"] = df["AdjustedVolume"] * df["SeasonalAdjustment"]

# 生成最终的交易量
df["FinalVolume"] = df["FinalVolume"].astype(int)

df[["Date", "FinalVolume"]].to_csv("data.csv")

# 查看生成的数据
print(df.head(20))

# 可视化交易量数据
plt.figure(figsize=(14, 7))
plt.plot(df["Date"], df["FinalVolume"], label="Simulated Trading Volume")
plt.title("Simulated Trading Volume Over Time")
plt.xlabel("Date")
plt.ylabel("Trading Volume")
plt.axvline(pd.to_datetime("2023-01-01"), color="r", linestyle="--", label="Holiday")
plt.axvline(pd.to_datetime("2023-05-01"), color="r", linestyle="--")
plt.axvline(pd.to_datetime("2023-10-01"), color="r", linestyle="--")
plt.axvline(pd.to_datetime("2023-12-25"), color="r", linestyle="--")
plt.legend()
plt.show()
plt.savefig("fig.png")
