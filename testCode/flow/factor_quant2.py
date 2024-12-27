import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 假设加载的数据如下
data = pd.DataFrame({
    'date': pd.date_range('2023-01-01', periods=100, freq='D'),
    'in': np.random.randint(1000, 5000, 100),
    'out': np.random.randint(500, 3000, 100),
    'cls': np.random.choice(['Category_A', 'Category_B', 'Category_C'], 100)
})

# 将日期列设为索引
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# 查看前几行数据
print(data.head())

# 计算净申购量
data['net_in_out'] = data['in'] - data['out']


# 模拟因子数据 (假设每日更新的3个因子)
factor_data = pd.DataFrame({
    'Factor_1': np.random.randn(100),  # 模拟因子1
    'Factor_2': np.random.randn(100),  # 模拟因子2
    'Factor_3': np.random.randn(100)   # 模拟因子3
}, index=data.index)

# 合并因子数据
data = pd.concat([data, factor_data], axis=1)

# 计算净申购量与因子之间的相关性
correlation_matrix = data[['net_in_out', 'Factor_1', 'Factor_2', 'Factor_3']].corr()
print(correlation_matrix)

import statsmodels.api as sm

# 准备回归分析数据
X = data[['Factor_1', 'Factor_2', 'Factor_3']]
X = sm.add_constant(X)  # 添加常数项
y = data['net_in_out']

# 拟合回归模型
model = sm.OLS(y, X).fit()

# 输出回归结果
print(model.summary())


# 因子1与净申购量的散点图
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Factor_1'], y=data['net_in_out'])
plt.title('Factor 1 vs Net Subscription/Redemption')
plt.xlabel('Factor 1')
plt.ylabel('Net Subscription/Redemption')
plt.show()


# 因子与净申购量的时间序列
plt.figure(figsize=(10, 6))
plt.plot(data.index, data['net_in_out'], label='Net Subscription/Redemption')
plt.plot(data.index, data['Factor_1'], label='Factor 1')
plt.plot(data.index, data['Factor_2'], label='Factor 2')
plt.plot(data.index, data['Factor_3'], label='Factor 3')
plt.title('Net Subscription/Redemption and Factors Over Time')
plt.xlabel('Date')
plt.ylabel('Value')
plt.legend()
plt.show()

# 热力图展示因子与净申购量的相关性
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Net Subscription/Redemption and Factors')
plt.show()


# 按类别分组并计算净申购量
category_groups = data.groupby('cls')

# 计算每个类别的平均净申购量
mean_net_in_out = category_groups['net_in_out'].mean()
print(mean_net_in_out)

# 进行 t 检验（例如，Category_A vs Category_B）
from scipy.stats import ttest_ind

category_a = data[data['cls'] == 'Category_A']['net_in_out']
category_b = data[data['cls'] == 'Category_B']['net_in_out']

t_stat, p_value = ttest_ind(category_a, category_b, nan_policy='omit')
print(f't-statistic: {t_stat}, p-value: {p_value}')


# 对in/out进行整体分析
# 计算滚动均值
data['in_roll'] = data['in'].rolling(window=7).mean()  # 7天滚动均值
data['out_roll'] = data['out'].rolling(window=7).mean()

# 绘制滚动均值
plt.figure(figsize=(12, 6))
plt.plot(data.index, data['in_roll'], label='Rolling Mean of Subscription (in)', color='blue')
plt.plot(data.index, data['out_roll'], label='Rolling Mean of Redemption (out)', color='red')
plt.title('7-Day Rolling Mean of Subscription and Redemption')
plt.xlabel('Date')
plt.ylabel('Amount')
plt.legend()
plt.show()


# 计算申购量和赎回量与因子的相关性
correlation_in = data[['in', 'Factor_1', 'Factor_2', 'Factor_3']].corr()
correlation_out = data[['out', 'Factor_1', 'Factor_2', 'Factor_3']].corr()

print("Correlation between Subscription (in) and Factors:")
print(correlation_in)

print("\nCorrelation between Redemption (out) and Factors:")
print(correlation_out)


# 可视化相关性热力图
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_in, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Subscription (in) and Factors')
plt.show()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_out, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Between Redemption (out) and Factors')
plt.show()

# 申购量与因子之间的回归分析
# 准备回归数据
X = data[['Factor_1', 'Factor_2', 'Factor_3']]
X = sm.add_constant(X)  # 添加常数项
y_in = data['in']

# 拟合回归模型
model_in = sm.OLS(y_in, X).fit()
print(model_in.summary())

# 赎回量与因子之间的回归分析
# 准备回归数据
y_out = data['out']

# 拟合回归模型
model_out = sm.OLS(y_out, X).fit()
print(model_out.summary())


# 按类别分组并计算每类基金的申购量和赎回量平均值
category_group_in = data.groupby('cls')['in'].mean()
category_group_out = data.groupby('cls')['out'].mean()

print("Average Subscription (in) by Category:")
print(category_group_in)

print("\nAverage Redemption (out) by Category:")
print(category_group_out)


# 选择两类基金的数据
category_a_in = data[data['cls'] == 'Category_A']['in']
category_b_in = data[data['cls'] == 'Category_B']['in']

# t 检验
from scipy.stats import ttest_ind
t_stat_in, p_value_in = ttest_ind(category_a_in, category_b_in, nan_policy='omit')
print(f't-statistic (Subscription in): {t_stat_in}, p-value: {p_value_in}')

# 赎回量的t检验
category_a_out = data[data['cls'] == 'Category_A']['out']
category_b_out = data[data['cls'] == 'Category_B']['out']

t_stat_out, p_value_out = ttest_ind(category_a_out, category_b_out, nan_policy='omit')
print(f't-statistic (Redemption out): {t_stat_out}, p-value: {p_value_out}')


# 方差分析（ANOVA）检验不同类别间申购量差异
from scipy.stats import f_oneway

category_a_in = data[data['cls'] == 'Category_A']['in']
category_b_in = data[data['cls'] == 'Category_B']['in']
category_c_in = data[data['cls'] == 'Category_C']['in']

# 方差分析
f_stat_in, p_value_in = f_oneway(category_a_in, category_b_in, category_c_in)
print(f'ANOVA result (Subscription in): F-statistic: {f_stat_in}, p-value: {p_value_in}')

# 赎回量的方差分析
category_a_out = data[data['cls'] == 'Category_A']['out']
category_b_out = data[data['cls'] == 'Category_B']['out']
category_c_out = data[data['cls'] == 'Category_C']['out']

f_stat_out, p_value_out = f_oneway(category_a_out, category_b_out, category_c_out)
print(f'ANOVA result (Redemption out): F-statistic: {f_stat_out}, p-value: {p_value_out}')
