import pandas as pd
def format():
    df = pd.read_csv("./origin/af_train_data.csv")
    df['HLA'] = df['HLA'].apply(lambda x: x[:-2] + ':' + x[-2:])
    df.to_csv("./origin/af_train_data.csv")
    
def merge():
    af_train_data = pd.read_csv('af_train_data.csv')
    
    # 读取IEDB_affinity_set.csv文件
    iedb_affinity_set = pd.read_csv('IEDB_affinity_set.csv')
    
    # 提取指定列
    af_train_data_subset = af_train_data[['Species', 'HLA', 'length', 'peptide', 'affinity', 'source']]
    iedb_affinity_set_subset = iedb_affinity_set[['Species', 'HLA', 'length', 'peptide', 'affinity', 'source']]
    
    # 合并数据
    merged_data = pd.concat([af_train_data_subset, iedb_affinity_set_subset], ignore_index=True)
    
    # 打印合并后的数据
    merged_data.to_csv("merge.csv")
    
def del_duplication():
    # 读取merge.csv文件
    merge_data = pd.read_csv('merge.csv')
    
    # 删除重复数据
    unique_data = merge_data.drop_duplicates(subset=['HLA', 'peptide'], keep='first')
    
    # 保存结果到ans.csv文件
    unique_data.to_csv('ans.csv', index=False)
    
    # 打印处理后的数据
    print("去重后的数据:")
    print(unique_data)
    
    

# merge()
del_duplication()