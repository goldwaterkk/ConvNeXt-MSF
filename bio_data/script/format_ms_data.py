import os
import pandas as pd
import numpy as np

def process_file_format():
    input_file = './origin/NetMHCpan41_ms.csv'
    df = pd.read_csv(input_file)
    
    # 在HLA列的第五个字符处添加"*"
    df['HLA'] = df['HLA'].apply(lambda x: x[:5] + '*' + x[5:])
    
    # 添加"length"列，保存"peptide"列的字符串长度
    df['length'] = df['peptide'].apply(len)
    
    # 删除"HLA_sequence"列
    df = df.drop(columns=['HLA_sequence'])
    
    # 保存到新的CSV文件
    output_file = 'NetMHCpan_format.csv'
    df.to_csv(output_file, index=False)
    
    print(f"文件已保存为 {output_file}")
    
def merge_csv():
    # 指定文件夹路径
    folder_path = 'origin'
    
    # 获取文件夹下所有文件
    all_files = os.listdir(folder_path)
    
    # 筛选出 CSV 文件
    csv_files = [file for file in all_files if file.endswith('.csv')]
    
    # 如果没有找到 CSV 文件，给出提示并退出
    if not csv_files:
        print("在指定文件夹下未找到CSV文件。")
        exit()
    
    # 创建一个空的 DataFrame 来存储合并后的数据
    merged_df = pd.DataFrame()
    
    # 循环读取并合并每个 CSV 文件
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        df = pd.read_csv(file_path)
        merged_df = pd.concat([merged_df, df], ignore_index=True)
    
    # 统计重复数据量
    duplicate_count = merged_df.duplicated(subset=['HLA', 'peptide']).sum()
    
    # 删除重复数据
    merged_df = merged_df.drop_duplicates(subset=['HLA', 'peptide'])
    
    # 保存合并后的数据到新的 CSV 文件
    output_file = 'merged_data_no_duplicates.csv'
    merged_df.to_csv(output_file, index=False)
    
    print(f"文件已保存为 {output_file}")
    print(f"删除了 {duplicate_count} 条重复数据。")
    



    
def del_negative_data():
    # 读取merged_data_no_duplicates.csv文件
    file_path = './merged_data_no_duplicates.csv'
    df = pd.read_csv(file_path)
    
    # 筛选label为0的行
    label_0_indices = df[df["label"] == 0].index
    
    # 计算label为1的行数和label为0的行数的差值
    difference = len(df[df["label"] == 1]) - len(df[df["label"] == 0])
    
    # 如果差值为正，表示label为1的行数多，随机删除label为1的行
    if difference > 0:
        random_indices_to_delete = np.random.choice(df[df["label"] == 1].index, size=difference, replace=False)
        df = df.drop(random_indices_to_delete)
    # 如果差值为负，表示label为0的行数多，随机删除label为0的行
    elif difference < 0:
        random_indices_to_delete = np.random.choice(label_0_indices, size=(abs(difference)), replace=False)
        df = df.drop(random_indices_to_delete)
    
    # 将结果保存为ans.csv
    df.to_csv("ans.csv", index=False)

def test_add():
    # 读取ms_train_data.csv文件
    df = pd.read_csv("ms_train_data.csv")
    
    # 在DataFrame中增加表头'9mer'和'affinity_netMHCpan'
    df["9mer"] = ""
    df["affinity_netMHCpan"] = ""
    
    # 将结果保存为新的CSV文件（如果需要）
    df.to_csv("ms_train_data_modified.csv", index=False)
    
    print("表头增加完成，结果已保存为ms_train_data_modified.csv。")


# merge_csv()
test_add()