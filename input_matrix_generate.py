import numpy as np
import concurrent.futures
from utils import *


def similarity_pseudo_seq(peptide, HLA):
    # 返回伪序列，生成输入矩阵要素之一
    # 针对输入的peptide和HLA按照Kmeans中的距离定义，寻找模板PDB，进而获得pseudo序列
    HLA = HLA.replace(":", "")  # 去除：号，与其他内容保持统一
    kmeans_data = read_HLA_kmeans_data()
    kmeans_HLA = kmeans_data[kmeans_data['HLA'] == HLA]
    # 获取这些行中的peptide列的值
    selected_peptides = kmeans_HLA['peptide']

    # 初始化最小得分和对应的peptide
    min_score = float('inf')  # 初始设为正无穷大
    min_score_peptide = None

    # 遍历selected_peptides的值
    for temple_peptide in selected_peptides:
        # 调用score函数计算得分
        current_score = non_negative_similarity_score(temple_peptide, peptide)  # 请替换score函数的调用方式
        # 检查当前得分是否小于最小得分
        if current_score < min_score:
            min_score = current_score
            min_score_peptide = temple_peptide

    # 组装HLA和temple_peptide
    HLA = HLA.replace("-", "_")  # 去除：号，与其他内容保持统一
    PDB_index = HLA + "_" + min_score_peptide + ".pdb"

    PDB_mapping_data = read_PDB_pseudo_site_mapping()
    pseudo_sequence = PDB_mapping_data[PDB_index]
    return pseudo_sequence


def get_one_matrix_AAindex(pseudo_sequence, peptide):
    # 将遍历pseudo sequence 同时结合 peptide，生成序列矩阵
    seq_matrix = []
    for pep_index, pep_pseudo_sequence in enumerate(pseudo_sequence):  # pep in [0-8]
        left_seq = pep_pseudo_sequence[2][0] + pep_pseudo_sequence[1][0] + pep_pseudo_sequence[0][0]
        right_seq = pep_pseudo_sequence[0][1] + pep_pseudo_sequence[1][1] + pep_pseudo_sequence[2][1]
        # 在index位置，组合伪序列和peptide
        seq_index = left_seq + peptide[pep_index] + right_seq
        seq_matrix.append(seq_index)

    AAindex_mapping = read_aaindex_value()  # mapping类型数据，第一索引为AAindex名字，第二索引为氨基酸变换数值
    float_matrix = []  # one matrix of X, shape = (5,9,7)
    for AAindex_type in AAindex_mapping.keys():  # 遍历所有key, max=5
        mapping_data = AAindex_mapping[AAindex_type]
        one_type_float_matrix = []  # 9*7
        for seq in seq_matrix:  # 遍历peptide, 第i位置(max = 9)对应的seq，转化为AAindex_type对应的float序列
            float_data = [mapping_data[char] for char in seq]  # 1*7
            one_type_float_matrix.append(float_data)  # (n+1)*7
        float_matrix.append(one_type_float_matrix)  # 5*9*7

    float_matrix = np.array(float_matrix)  # 转化为numpy数据
    float_matrix = np.transpose(float_matrix, (1, 2, 0))  # 转换为维度(9,7,5)
    # 之后映射为AAindex矩阵
    return float_matrix  # type : numpy , shape: (9,7,5)


def generate_ms_data(file_path, save_path):
    # (1) 根据输入的file_path,读取9mer和HLA
    # (2) 由函数计算评分，获得残基序列，加上9mer，生成序列矩阵
    df_ms_data = pd.read_csv(file_path)
    # 遍历每一行
    input_matrix = []
    input_label = []
    index = 0
    for index, row in df_ms_data.iterrows():
        # 访问HLA列和9mer列的值
        hla_value = row['HLA']
        peptide_value = row['9mer']
        label_value = row['label']
        pseudo_sequence = similarity_pseudo_seq(peptide_value, hla_value)
        one_input_matrix = get_one_matrix_AAindex(pseudo_sequence, peptide_value)
        one_label = np.array(label_value)
        input_matrix.append(one_input_matrix)
        input_label.append(one_label)
        index += 1

    np.savez(save_path, x=input_matrix, y=input_label)


def read_ms_data(save_path):
    data = np.load(save_path)
    x_data = data["x"]
    y_data = data["y"]
    print(x_data.shape)
    print(y_data.shape)


def main():
    # 使用上下文管理器创建线程池
    with concurrent.futures.ThreadPoolExecutor() as executor:
        # 提交每个任务到线程池
        futures = [executor.submit(process_fold, i) for i in range(20)]

        # 等待所有任务完成
        concurrent.futures.wait(futures)


def process_fold(i):
    ms_file_path = "./data/csv/k_folds/ms_fold_{}.csv".format(i)
    ms_save_path = "./data/npy/k_folds/ms_fold_{}.npz".format(i)
    generate_ms_data(ms_file_path, ms_save_path)


# 调用主函数
main()
