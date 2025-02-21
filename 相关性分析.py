import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.stattools import durbin_watson

# 文件路径
file_path = r"D:\C_data\消费决策.csv"

# 读取文件，指定编码格式为 GBK
data = pd.read_csv(file_path, header=None, encoding='gbk')

# 为数据添加列名
data.columns = ['y', 'X1', 'X2', 'X3', 'X4', 'X5']

# 拟合多元线性回归模型
model = ols('y ~ X1 + X2 + X3 + X4 + X5', data=data).fit()

# 获取方差分析表
anova_table = anova_lm(model, typ=2)  # 使用 Type II ANOVA

# 打印方差分析表
print("方差分析表：")
print(anova_table)

# 单独提取回归平方和、残差平方和、自由度和均方
regression_ss = model.ssr  # 回归平方和 (Sum of Squares Regression)
residual_ss = model.centered_tss - model.ssr  # 残差平方和 (Sum of Squares Error)
regression_df = model.df_model  # 回归自由度 (Degrees of Freedom Regression)
residual_df = model.df_resid  # 残差自由度 (Degrees of Freedom Error)
regression_ms = regression_ss / regression_df  # 回归均方 (Mean Square Regression)
residual_ms = residual_ss / residual_df  # 残差均方 (Mean Square Error)

# 获取 F 统计量和对应的 p 值
f_statistic = model.fvalue
f_pvalue = model.f_pvalue

print(f"F 统计量: {f_statistic:.4f}")
print(f"F 检验的 p 值: {f_pvalue:.4e}")

# 打印详细结果
print("\n详细结果：")
print(f"回归平方和（SSR）：{regression_ss:.4f}")
print(f"残差平方和（SSE）：{residual_ss:.4f}")
print(f"回归自由度（dfR）：{regression_df}")
print(f"残差自由度（dfE）：{residual_df}")
print(f"回归均方（MSR）：{regression_ms:.4f}")
print(f"残差均方（MSE）：{residual_ms:.4f}")

# 计算 R、R方、调整后 R方
r_squared = model.rsquared
adj_r_squared = model.rsquared_adj

print(f"\nR 方: {r_squared:.4f}")
print(f"调整后 R 方: {adj_r_squared:.4f}")

# 计算得宾-沃森统计量
residuals = model.resid
dw_statistic = durbin_watson(residuals)

print(f"\n得宾-沃森统计量: {dw_statistic:.4f}")

# 获取未标准化系数、标准误差、t 值和 p 值
params = model.params
b_values = params[1:]  # 未标准化系数 (B)
standard_errors = model.bse[1:]  # 标准误差
t_values = model.tvalues[1:]  # t 值
p_values = model.pvalues[1:]  # p 值

# 计算标准化系数
standardized_coefficients = model.params[1:] * (data.std()[1:] / data['y'].std())

# 打印未标准化系数、标准误差、标准化系数、t 值和 p 值
print("\n回归系数表：")
for i in range(1, 6):
    print(f"X{i}:")
    print(f"  未标准化系数 (B): {b_values[i-1]:.4f}")
    print(f"  标准误差: {standard_errors[i-1]:.4f}")
    print(f"  标准化系数: {standardized_coefficients[i-1]:.4f}")
    print(f"  t 值: {t_values[i-1]:.4f}")
    print(f"  p 值: {p_values[i-1]:.4e}")

# 计算容差和 VIF
# 构建自变量的相关性矩阵
X_data = data[['X1', 'X2', 'X3', 'X4', 'X5']]
correlation_matrix_X = X_data.corr()

# 计算容差矩阵
tolerance_matrix = pd.DataFrame(np.linalg.inv(correlation_matrix_X.values), index=correlation_matrix_X.columns, columns=correlation_matrix_X.columns)

# 提取容差和 VIF
tolerance = tolerance_matrix['X1']  # 容差
vif = 1 / tolerance  # VIF

print("\n容差和 VIF：")
for i in range(1, 6):
    print(f"X{i}:")
    print(f"  容差: {tolerance[i-1]:.4f}")
    print(f"  VIF: {vif[i-1]:.4f}")

# 计算 y 与每个自变量的皮尔逊相关系数
correlation_matrix = data.corr().round(3)
print("\n相关性矩阵：")
print(correlation_matrix)

# 单独计算 y 与每个自变量的相关性
for i in range(1, 6):
    x_column = data[f'X{i}']  # 获取第 i 列作为自变量
    correlation, p_value = pearsonr(data['y'], x_column)  # 计算相关系数和 p 值
    print(f"\ny 与 X{i} 的皮尔逊相关系数：{correlation:.3f}, p 值：{p_value:.4e}")
    
# 计算自变量之间的相关系数及其 p 值
for i in range(1, 6):
    for j in range(i + 1, 6):
        x_col_i = data[f'X{i}']
        x_col_j = data[f'X{j}']
        corr_ij, p_value_ij = pearsonr(x_col_i, x_col_j)
        print(f"自变量 X{i} 和 X{j} 的皮尔逊相关系数：{corr_ij:.3f}, p 值：{p_value_ij:.4e}")
