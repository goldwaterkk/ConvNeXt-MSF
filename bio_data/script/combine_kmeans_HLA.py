import os
import pandas as pd

# 指定文件夹路径
folder_path = 'kmeans_for_HLA'

# 获取文件夹下的所有CSV文件
csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv')]

# 初始化一个空的DataFrame来存储合并后的数据
merged_data = pd.DataFrame()

# 循环读取每个CSV文件并合并到DataFrame中
for csv_file in csv_files:
    file_path = os.path.join(folder_path, csv_file)
    df = pd.read_csv(file_path)
    merged_data = pd.concat([merged_data, df], ignore_index=True)

# 将合并后的数据保存到新的CSV文件中
merged_data.to_csv('merged_data.csv', index=False)

print("合并完成并保存到 merged_data.csv 文件中。")