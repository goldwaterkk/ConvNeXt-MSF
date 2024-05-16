import pandas as pd

# 读取CSV文件，指定第一行为表头
df = pd.read_csv('aaindex1.csv', header=0)

# 提取列名
columns = df.columns

# 对每一行（除去索引列和表头）进行线性变换
df_normalized = df.iloc[:, 1:].apply(lambda row: (row - row.min()) / (row.max() - row.min()), axis=1)

# 合并索引列、表头和变换后的数据
df_normalized = pd.concat([df[columns[0]], df_normalized], axis=1)

# 打印变换后的数据
print(df_normalized)


df_normalized.to_csv("format.csv")