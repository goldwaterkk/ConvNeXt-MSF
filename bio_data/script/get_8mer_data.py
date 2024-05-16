import pandas as pd
from Bio import pairwise2
from sklearn.cluster import KMeans
import seaborn as sns
import matplotlib.pyplot as plt
from Bio.SubsMat import MatrixInfo
import os
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 读取原始数据
def divid_origin():
    df = pd.read_csv('ms_train_data.csv')
    
    # 根据length列分割数据
    length_9_df = df[df['length'] == 9]
    non_length_9_df = df[df['length'] != 9]
    
    # 保存为新的CSV文件
    length_9_df.to_csv('9mer.csv', index=False)
    non_length_9_df.to_csv('non9mer.csv', index=False)
    
def count_9merdata():
    df = pd.read_csv('ms_train_data.csv')
    
    # 统计"HLA"列的非重复元素数量
    unique_hla_count = df['HLA'].nunique()
    print(unique_hla_count)
    
    # 统计每个不同的HLA值对应的行数
    hla_counts = df['HLA'].value_counts()
    
    # 输出结果
    print("每个HLA值对应的行数:")
    print(hla_counts[-30:])
    
    
def non_negative_similarity_score(sequence1, sequence2, substitution_matrix=MatrixInfo.blosum62, offset=10):
    """
    计算两个氨基酸序列的非负相似性得分

    参数:
    - sequence1: 第一个氨基酸序列
    - sequence2: 第二个氨基酸序列
    - substitution_matrix: 替代矩阵，默认为 BLOSUM62
    - offset: 偏移量，用于确保得分非负，默认为 10

    返回:
    - 非负相似性得分
    """
    if len(sequence1) != len(sequence2):
        raise ValueError("序列长度不一致")

    min_score = min(substitution_matrix.values())
    score = 0
    for aa1, aa2 in zip(sequence1, sequence2):
        if (aa1, aa2) in substitution_matrix:
            score += substitution_matrix[(aa1, aa2)]
        else:
            score += substitution_matrix[(aa2, aa1)]  # 考虑反向顺序

    return max(score + offset - min_score, 0)

def get_HLA_to_peptide2(HLA = "HLA-B13:01"):
    df = pd.read_csv('9mer.csv')
    selected_rows = df[df['HLA'] == HLA]
    peptide_data = selected_rows['peptide'].to_numpy()
    return peptide_data


def kmeans():
    # 给定的序列字符串数组
    get_HLA_to_peptide = get_HLA_to_peptide2()   
    # 计算两两序列之间的距离
    num_sequences = len(get_HLA_to_peptide)
    distance_matrix = [[0] * num_sequences for _ in range(num_sequences)]
    
    for i in range(num_sequences):
        for j in range(i+1, num_sequences):
            distance_matrix[i][j] = non_negative_similarity_score(get_HLA_to_peptide[i], get_HLA_to_peptide[j])
            distance_matrix[j][i] = distance_matrix[i][j]
    
    # 使用k-means聚类
    kmeans = KMeans(n_clusters=3, random_state=42)
    clusters = kmeans.fit_predict(distance_matrix)
    
    # 绘制聚类结果的热图
    sns.heatmap(distance_matrix, cmap='viridis', annot=True, fmt=".2f", xticklabels=get_HLA_to_peptide, yticklabels=get_HLA_to_peptide)
    plt.title('Sequence Distance Heatmap')
    plt.show()
    
    # 输出聚类结果
    print("聚类结果：", clusters)
    
def divid_by_HLA():
    # 读取CSV文件
    csv_file_path = '9mer.csv'
    df = pd.read_csv(csv_file_path)
    
    # 按照HLA列进行划分
    grouped = df.groupby('HLA')
    
    # 创建保存文件的目录
    output_directory = './divid_by_HLA'
    os.makedirs(output_directory, exist_ok=True)
    
    # 遍历每个分组，将其保存到相应的文件中
    for hla_value, group_df in grouped:
        # 生成保存文件的路径
        output_file_path = os.path.join(output_directory, f'{hla_value}.csv')
        
        # 将分组数据保存到文件
        group_df.to_csv(output_file_path, index=False)
    
    print("文件已成功保存到 ./divid_by_HLA 目录中。")
    
def kmeans_record():
    # 创建保存聚类结果的目录
    output_directory = './kmeans_for_HLA'
    os.makedirs(output_directory, exist_ok=True)
    
    # 遍历每个HLA文件
    for filename in os.listdir('./divid_by_HLA'):
        if filename.endswith(".csv"):
            filepath = os.path.join('./divid_by_HLA', filename)
            
            # 读取CSV文件
            df = pd.read_csv(filepath)
            
            # 计算相似度矩阵
            peptides = df['peptide'].tolist()
            similarity_matrix = [[non_negative_similarity_score(p1, p2) for p2 in peptides] for p1 in peptides]
    
            # 使用KMeans进行聚类
            num_clusters = len(df) // 100
            kmeans = KMeans(n_clusters=num_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(StandardScaler().fit_transform(similarity_matrix))
    
            # 添加聚类结果到DataFrame
            df['cluster'] = cluster_labels
    
            # 添加cluster_kernel列
            df['cluster_kernel'] = 0
            for cluster_center in kmeans.cluster_centers_:
                center_index = np.argmin(np.sum((StandardScaler().fit_transform(similarity_matrix) - cluster_center) ** 2, axis=1))
                df.at[center_index, 'cluster_kernel'] = 1
    
            # 保存聚类结果到新的文件
            output_filepath = os.path.join(output_directory, filename)
            df.to_csv(output_filepath, index=False)
    
    print("聚类结果已成功保存到 ./kmeans_for_HLA 目录中。")
    
def kmeans_record2():

    # 创建保存聚类结果的目录
    output_directory = './kmeans_for_HLA'
    os.makedirs(output_directory, exist_ok=True)
    
    # 遍历每个HLA文件
    for filename in os.listdir('./divid_by_HLA'):
        if filename.endswith(".csv"):
            try:
                filepath = os.path.join('./divid_by_HLA', filename)
                
                # 读取CSV文件
                df = pd.read_csv(filepath)
                
                # 计算相似度矩阵
                peptides = df['peptide'].tolist()
                similarity_matrix = [[non_negative_similarity_score(p1, p2) for p2 in peptides] for p1 in peptides]
        
                # 使用KMeans进行聚类
                num_clusters = len(df) // 100 + 1
                if num_clusters > 20:
                    num_clusters = 20
                kmeans = KMeans(n_clusters=num_clusters, random_state=42)
                cluster_labels = kmeans.fit_predict(StandardScaler().fit_transform(similarity_matrix))
        
                # 添加聚类结果到DataFrame
                df['cluster'] = cluster_labels
        
                # 添加cluster_kernel列
                df['cluster_kernel'] = 0
                for cluster_center in kmeans.cluster_centers_:
                    center_index = np.argmin(np.sum((StandardScaler().fit_transform(similarity_matrix) - cluster_center) ** 2, axis=1))
                    df.at[center_index, 'cluster_kernel'] = 1
        
                # 保存聚类结果到原始文件
                df.to_csv(filepath, index=False)
        
                # 找到与每个聚类中心最近的peptide所在行的数据
                nearest_indices = [np.argmin(np.sum((StandardScaler().fit_transform(similarity_matrix) - center_index) ** 2, axis=1))
                                   for center_index in kmeans.cluster_centers_]
                
                # 保存结果
                output_filename = f"kmeans_result_{filename}"
                output_filepath = os.path.join(output_directory, output_filename)
                df.iloc[nearest_indices].to_csv(output_filepath, index=False)
            except Exception:
                continue

# 149 HLA
# kmeans()
# kmeans()
# count_9merdata()
kmeans_record2()