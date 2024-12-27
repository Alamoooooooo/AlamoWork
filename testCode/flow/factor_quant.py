import pandas as pd
import numpy as np
from scipy.stats import pearsonr, ttest_ind
import statsmodels.api as sm

# 固定随机种子
np.random.seed(42)

# 假设 T = 100（时间长度），N = 5（基金数量），M = 3（因子数量）
T, N, M = 100, 5, 3

# 模拟申赎量数据（单位：百万份）
subscriptions = pd.DataFrame(np.random.randn(T, N) * 10 + 50, columns=[f"Fund_{i+1}" for i in range(N)])

# 模拟因子数据
factors = pd.DataFrame(np.random.randn(T, M), columns=[f"Factor_{j+1}" for j in range(M)])
print(factors)

# 计算相关性
def compute_correlation(subscriptions, factors, lag=0):
    correlations = {}
    for factor in factors.columns:
        factor_lagged = factors[factor].shift(lag)  # 滞后因子
        for fund in subscriptions.columns:
            valid_idx = ~subscriptions[fund].isna() & ~factor_lagged.isna()  # 有效数据
            corr, _ = pearsonr(subscriptions[fund][valid_idx], factor_lagged[valid_idx])
            correlations[(fund, factor)] = corr
    return pd.DataFrame.from_dict(correlations, orient='index', columns=['Correlation'])

# 示例：计算相关性
correlation_matrix = compute_correlation(subscriptions, factors, lag=1)
print(correlation_matrix)


# 回归分析
def perform_regression(subscriptions, factors, lag=0):
    results = {}
    for fund in subscriptions.columns:
        Y = subscriptions[fund].dropna()
        X = factors.shift(lag).loc[Y.index].dropna()
        X = sm.add_constant(X)  # 添加截距项
        if len(Y) == len(X):  # 确保对齐
            model = sm.OLS(Y, X).fit()
            results[fund] = model.params
    return pd.DataFrame(results).T

# 示例：进行回归分析
regression_results = perform_regression(subscriptions, factors, lag=1)
print(regression_results)


# 分组并检验差异
def group_and_test(subscriptions, factors, factor_name, groups=5):
    factor = factors[factor_name]
    group_test_results = {}
    for fund in subscriptions.columns:
        # 因子排序分组
        factor_sorted = factor.sort_values()
        group_labels = pd.qcut(factor_sorted, groups, labels=False)
        
        # 分组后计算申赎量均值
        grouped_subscriptions = subscriptions[fund].groupby(group_labels).mean()
        
        # 差异性 t 检验（最低组 vs 最高组）
        t_stat, p_value = ttest_ind(
            subscriptions[fund][group_labels == 0],
            subscriptions[fund][group_labels == groups - 1],
            nan_policy='omit'
        )
        group_test_results[fund] = {'t_stat': t_stat, 'p_value': p_value}
    return pd.DataFrame(group_test_results).T

# 示例：对 Factor_1 分组并检验申赎量差异
difference_test_results = group_and_test(subscriptions, factors, factor_name="Factor_1", groups=5)
print(difference_test_results)
