import pandas as pd
import numpy as np
import timly_baselines.holidays as holidays
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.seasonal import seasonal_decompose
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import timedelta

# 1. 加载数据
data = pd.read_csv("data.csv", parse_dates=["Date"])[["Date", "FinalVolume"]]
data.columns = ["DATE", "bal"]
data["DATE"] = pd.to_datetime(data["DATE"])

# 确保数据按照日期排序
data.set_index("DATE", inplace=True)

# 设定预测步长，分割训练集和测试集
PRED_STEP_LEN = 60
sd = "2023-09-15"
sdd = (pd.to_datetime(sd) + timedelta(days=1)).strftime("%Y-%m-%d")
ed = (pd.to_datetime(sd) + timedelta(days=PRED_STEP_LEN)).strftime("%Y-%m-%d")
train_data = data[:sd]
test_data = data[sdd:ed]

# 打印训练集和测试集的起止日期以验证
print(f"Train data range: {train_data.index.min()} to {train_data.index.max()}")
print(f"Test data range: {test_data.index.min()} to {test_data.index.max()}")

# 获得中国大陆历法节日列表，并转换成list类型或{节日：日期}的字典类型
cn_holidays = holidays.CN()
holidays_list = list(cn_holidays.keys())
holidays_dict = {date: holiday for holiday, date in cn_holidays.items()}

# 获得节假日的日期列表
holiday_dates = [holidays_dict[holiday] for holiday in holidays_list]

# 1. 创建节日特征
# holidays = ["2023-12-25", "2024-01-01"]  # 手动指定节假日
holidays = holiday_dates

train_data.loc[:, "is_holiday"] = train_data.index.isin(holidays).astype(
    int
)  # 转换为int类型
test_data.loc[:, "is_holiday"] = test_data.index.isin(holidays).astype(
    int
)  # 对测试集同样处理

# # 提取外生变量（节日特征）
# exog_train = train_data[["is_holiday"]]
# exog_test = test_data[["is_holiday"]]

# 调整每个节日日期的影响强度与范围参数
holiday_influence_days = 3  # 节日影响前后3天


# 创建节日影响的特征
def create_holiday_feature(df, holidays, influence_days):
    df["holiday_weight"] = 0  # 初始影响为0
    for holiday in holidays:
        # 设置节日及其前后几天的影响范围
        holiday_range = pd.date_range(
            start=holiday - timedelta(days=influence_days),
            end=holiday + timedelta(days=influence_days),
        )
        df.loc[df.index.isin(holiday_range), "holiday_weight"] = 5
    return df


# 在训练集和测试集中添加节日影响特征
train_data = create_holiday_feature(train_data, holidays, holiday_influence_days)
test_data = create_holiday_feature(test_data, holidays, holiday_influence_days)

# 提取外生变量（节日特征）
exog_train = train_data[["holiday_weight"]]
exog_test = test_data[["holiday_weight"]]

# 2. 使用SARIMAX模型
model_sarimax = SARIMAX(
    train_data["bal"], order=(4, 0, 2), seasonal_order=(52, 1, 4, 7), exog=exog_train
)
result_sarimax = model_sarimax.fit()

# 3. 进行预测
sarimax_forecast = result_sarimax.get_forecast(steps=PRED_STEP_LEN, exog=exog_test)

# 提取预测值
final_forecast_sarimax = sarimax_forecast.predicted_mean

# 4. 绘制SARIMAX预测结果
plt.figure(figsize=(10, 6))
plt.plot(
    test_data.index, final_forecast_sarimax, label="SARIMAX Prediction", color="green"
)
plt.plot(test_data.index, test_data["bal"], label="True Values", color="blue")
plt.title("SARIMAX Prediction vs True Values")
plt.xlabel("Date")
plt.ylabel("Value")
plt.legend()
plt.grid(True)
plt.show()

# 4. 计算并打印MAE和MSE
mae = mean_absolute_error(test_data["bal"], final_forecast_sarimax)
mse = mean_squared_error(test_data["bal"], final_forecast_sarimax)
print(f"MAE: {mae}")
print(f"MSE: {mse}")
