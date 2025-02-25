import pandas as pd
import numpy as np
from factor_analyzer import FactorAnalyzer, calculate_kmo
from scipy.stats import bartlett

# ========== 数据读取 ==========
data = pd.read_csv('D:/C_data/app_choices.csv', header=None)
data.columns = [f'变量{i+1}' for i in range(data.shape[1])]  # 根据实际列数动态生成列名

# ========== 数据预处理 ==========
print("数据预览：")
print(data.head().round(3))  # 数据预览保留三位小数
print("\n数据描述统计：")
print(data.describe().round(3))  # 描述统计保留三位小数

# ========== 适用性检验 ==========
kmo_all, kmo_value = calculate_kmo(data)
chi_square, p_value = bartlett(*data.T.values)

# ========== 初始因子分析 ==========
fa_initial = FactorAnalyzer(rotation=None, method='principal', n_factors=data.shape[1])
fa_initial.fit(data)
ev_initial, _ = fa_initial.get_eigenvalues()

# ========== 完整方差解释率表格 ==========
variance_total_initial, variance_prop_initial, variance_cum_initial = fa_initial.get_factor_variance()
variance_df_initial = pd.DataFrame({
    '特征值': variance_total_initial,
    '方差解释率(%)': (variance_prop_initial * 100).round(3),
    '累积方差解释率(%)': (variance_cum_initial * 100).round(3)
}, index=[f'因子{i+1}' for i in range(data.shape[1])])

# ========== 修改特征根阈值 ==========
threshold = 1.0  # 可以在此处自由调整阈值
num_factors_initial = sum(ev_initial > threshold)

# ========== 计算自由度（df） ==========
n_variables = data.shape[1]  # 变量数量
n_factors = num_factors_initial  # 提取的因子数量
df = (n_variables * (n_variables - 1) // 2) - (n_factors * (n_factors - 1) // 2)

# ========== 主因子分析 ==========
fa = FactorAnalyzer(n_factors=num_factors_initial, rotation='varimax', method='principal')
fa.fit(data)

# ========== 方差解释率 ==========
variance_total, variance_prop, variance_cum = fa.get_factor_variance()
variance_df = pd.DataFrame({
    '特征值': variance_total,
    '方差解释率(%)': (variance_prop * 100).round(3),
    '累积方差解释率(%)': (variance_cum * 100).round(3)
}, index=[f'因子{i+1}' for i in range(num_factors_initial)])

# ========== 因子载荷矩阵 ==========
loadings_df = pd.DataFrame(
    fa.loadings_,
    index=data.columns,
    columns=[f'因子{i+1}' for i in range(num_factors_initial)]
).round(3)  # 保留三位小数

# ========== 成分得分系数矩阵 ==========
corr_matrix = data.corr()
inv_corr = np.linalg.inv(corr_matrix)
score_coefficients = pd.DataFrame(
    inv_corr @ fa.loadings_,
    index=data.columns,
    columns=[f'因子{i+1}' for i in range(num_factors_initial)]
).round(3)  # 保留三位小数

# ========== 修正后的关键计算部分 ==========
linear_combination = loadings_df * variance_prop.reshape(1, -1)
cumulative_variance = variance_cum[-1]
composite_scores = linear_combination.sum(axis=1) / cumulative_variance
weights_final = composite_scores / composite_scores.sum()

# ========== 结果整合 ==========
results_df = pd.DataFrame({
    **{f'因子{i+1}': loadings_df.iloc[:, i] for i in range(num_factors_initial)},
    '综合得分系数': composite_scores.round(3),
    '权重': weights_final.round(3)
}, index=data.columns)

# ========== 计算共同度（公因子方差） ==========
communalities = fa.get_communalities()
communalities_df = pd.DataFrame(communalities, index=data.columns, columns=['共同度(%)']).round(3)  # 保留三位小数

# ========== 打印结果 ==========
print("="*50)
print(f"KMO检验值：{kmo_value:.3f}")
print(f"巴特利特检验P值：{p_value:.3e}")
print(f"巴特利特检验近似卡方值：{chi_square:.3f}")
print(f"巴特利特检验自由度（df）：{df}")
print(f"建议保留因子数量（特征根>{threshold}）：{num_factors_initial}")
print("="*50)
print("完整方差解释率表格（所有因子）：")
print(variance_df_initial)
print("\n筛选后的方差解释率表格（特征根>{threshold}）：")
print(variance_df)
print("\n旋转后的因子载荷矩阵：")
print(loadings_df)
print("\n共同度（公因子方差）：")
print(communalities_df)
print("\n成分得分系数矩阵：")
print(score_coefficients)
print("\n综合得分系数及权重结果：")
print(results_df)

# ========== 将结果写入Excel文件 ==========
excel_file = '因子分析结果.xlsx'
with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
    # 1. 数据预览
    data.head(5).round(3).to_excel(writer, sheet_name='数据预览', index=False)
    
    # 2. 描述统计
    data.describe().round(3).to_excel(writer, sheet_name='描述统计', index=True)
    
    # 3. 检验结果
    test_results = pd.DataFrame({
        '检验指标': ['KMO检验值', '巴特利特检验P值', '巴特利特检验卡方值', '巴特利特自由度', '建议因子数'],
        '结果': [f"{kmo_value:.3f}", 
                f"{p_value:.3e}", 
                f"{chi_square:.3f}",
                df,
                num_factors_initial]
    })
    test_results.to_excel(writer, sheet_name='检验结果', index=False)
    
    # 4. 方差解释率表
    variance_df_initial.to_excel(writer, sheet_name='完整方差解释', index=True)
    variance_df.to_excel(writer, sheet_name='筛选方差解释', index=True)
    
    # 5. 因子载荷矩阵
    loadings_df.to_excel(writer, sheet_name='因子载荷矩阵', index=True)
    
    # 6. 共同度
    communalities_df.to_excel(writer, sheet_name='共同度', index=True)
    
    # 7. 成分得分系数
    score_coefficients.to_excel(writer, sheet_name='成分得分系数', index=True)
    
    # 8. 综合得分及权重
    results_df.to_excel(writer, sheet_name='综合得分权重', index=True)

print(f"\n所有结果已成功保存至 {excel_file}")
