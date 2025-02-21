import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder

# 1. 读取CSV文件
file_path = r'D:\C_data\效用估计值.csv'  # 使用原始字符串
data = pd.read_csv(file_path, header=None)

# 2. 数据预处理
data.columns = ['Q13', 'Q14', 'Q15', 'Q17']

# 3. 创建虚拟变量（One-Hot Encoding）
encoder = OneHotEncoder(sparse_output=False)  # 不使用稀疏矩阵
encoded_data = encoder.fit_transform(data)

# 获取列名
encoded_columns = encoder.get_feature_names_out(input_features=data.columns)

# 将编码后的数据转换为DataFrame
df_encoded = pd.DataFrame(encoded_data, columns=encoded_columns)

# 4. 构建回归模型
data['TotalUtility'] = data.sum(axis=1)  # 假设效用是四个属性的总和

X = df_encoded  # 使用虚拟变量作为特征
y = data['TotalUtility']  # 目标变量

# 训练线性回归模型
reg_model = LinearRegression()
reg_model.fit(X, y)

# 5. 提取系数作为效用估计值
coefficients = reg_model.coef_
intercept = reg_model.intercept_

# 创建效用估计值的DataFrame
utility_estimates = pd.DataFrame({
    'Feature': df_encoded.columns,
    'Utility_Estimate': coefficients
})

# 设置 Pandas 显示选项，避免科学计数法
pd.set_option('display.float_format', lambda x: '%.3f' % x)

# 打印效用估计值
print("效用估计值：")
print(utility_estimates)
