import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# 读取数据
file_path = 'D:/C_data/user_choices.csv'
data = pd.read_csv(file_path)
data.columns = [f'Attribute{i}' for i in range(1, 9)]

# 独热编码
encoder = OneHotEncoder(sparse_output=False)
encoded_data = encoder.fit_transform(data)
encoded_columns = encoder.get_feature_names_out()
df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)

# 生成目标变量
data['TotalUtility'] = data.sum(axis=1)

# 回归模型
X = df_encoded
y = data['TotalUtility']
reg_model = LinearRegression()
reg_model.fit(X, y)

# 提取效用估计值
coefficients = reg_model.coef_
utility_estimates = pd.DataFrame({
    'Feature': df_encoded.columns,
    'Utility_Estimate': coefficients
})
pd.set_option('display.float_format', lambda x: '{:.3f}'.format(x))
print("效用估计值：")
print(utility_estimates)

# 计算属性重要性
attributes = [col.split('_')[0] for col in df_encoded.columns]
importance = pd.Series(coefficients**2, index=df_encoded.columns)
grouped_importance = importance.groupby(attributes).sum()
normalized_importance = grouped_importance / grouped_importance.sum()
attribute_importance = pd.DataFrame({
    'Attribute': grouped_importance.index,
    'Importance': normalized_importance.values
})
print("属性重要性：")
print(attribute_importance.sort_values('Importance', ascending=False))

# 查看模型的R²值
r2 = reg_model.score(X, y)
print(f"模型的R²值为：{r2:.3f}")
