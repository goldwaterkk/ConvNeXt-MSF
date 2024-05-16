import os
import pandas as pd
import json
from Bio.PDB import PDBParser
three_to_one = {
    'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
    'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
    'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
    'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V'
}

def calculate_distance(atom1, atom2):
    # 计算两个原子之间的欧几里得距离
    return atom1 - atom2

def distance_to_site(close_residue_list=[]):
    # 假设close_residue_list是一个包含字典的列表
    # 将数据转换为DataFrame
    df = pd.DataFrame(close_residue_list)

    # 根据distance属性分组
    grouped = df.groupby(pd.cut(df['distance'], bins=[0, 6.5, 7.5, float('inf')]))

    # 遍历每个分组，输出每组中distance最小的两个值对应的residue_type
    result_site = [] # 4-6,6-7,7-inf
    for group_name, group_df in grouped:
        min_distances = group_df.nsmallest(2, 'distance')
        output_residue_types = min_distances['residue_type'].tolist()
        result_site.append(output_residue_types)

    return result_site

def find_close_residues(chain, all_residues, threshold):
    close_residues = {}

    # 遍历链条中的所有氨基酸
    for residue1 in chain:
        # 初始化当前氨基酸的附近氨基酸列表
        close_residues[residue1.id[1]] = []

        # 获取当前氨基酸的α碳原子
        atom1 = residue1['CA']

        # 遍历所有氨基酸
        for residue2 in all_residues:
            # 获取目标氨基酸的α碳原子
            atom2 = residue2['CA']

            # 计算原子之间的距离
            distance = calculate_distance(atom1, atom2)

            # 如果距离小于阈值，则将氨基酸添加到结果集
            if distance < threshold:
                close_residues[residue1.id[1]].append({
                    'residue_id': residue2.id[1],
                    'residue_type': residue2.get_resname(),
                    'distance': distance
                })

    return close_residues

# 读取PDB文件
def get_PDB_residure_site(PDB_file = 'HLA_B3503_LPSSIVQML.pdb'):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure('protein_structure', './format_structure_data/'+PDB_file)
    
    # 选择蛋白质链条P
    selected_chain_id = 'P'
    
    # 设置距离阈值
    distance_threshold = 9.0
    
    # 获取指定链条
    chain_P = structure[0][selected_chain_id]
    
    # 获取所有氨基酸，排除链P中的氨基酸
    all_residues = []
    for chain_id in ['M']:
        all_residues.extend([residue for residue in structure[0][chain_id]])
        
    # 寻找距离小于4.0A的氨基酸
    close_residues = find_close_residues(chain_P, all_residues, distance_threshold)
    
    # 将具体内容转化为位置矩阵；
    '''
    7.5< x <=9.0     'G'     'G'
    6.5< x <=7.5     'A'     'G'
    0< x <=6.5     'L'     'G'
      pep1      null   null
    0< x <=6.5     'T'     'G'
    6.5< x <=7.5     'X'     'G'
    7.5< x <=9.0     'L'     'G'

    # 输出结果，6*9的string
    以前两个残基为例，string 表示
    保存六行string，每行a[] string保存九肽对应位置距离对应的残基氨基酸，形成6*9的矩阵a
    不保存peptide，直接将peptide对应的string插入到该矩阵中，形成7*9的结构指纹
    
    # 转换结果后续处理逻辑
    利用PCA获得的理化性质，直接将氨基酸化为7*9*3输入矩阵
    
    '''
    
    # 遍历9肽
    pesudo_site = []
    for residue_id, close_residue_list in close_residues.items():
        residue = chain_P[residue_id]
        # 在i个peptide情况下，遍历周围的所有氨基酸，取分级对应的最小两个值
        site = distance_to_site(close_residue_list)
        # 将氨基酸化为简写,如果长度不满足2，则添加空位X
        # site = [[three_to_one[aa] for aa in pair] for pair in site]
        site = [[three_to_one[aa] if aa in three_to_one else 'X' for aa in pair] + ['X'] * (2 - len(pair)) for pair in site]
        pesudo_site.append(site)
    return pesudo_site

    # return pesudo_site
    # [['W', 'C'], ['N', 'L'], ['L', 'G']]
    # [[], ['Y', 'N'], ['I', 'G']]
    # [[], ['Y'], ['L', 'I']]
    # [['I'], ['N'], ['F', 'T']]
    # [[], ['N'], ['T', 'V']]
    # [[], ['V'], ['A', 'Q']]
    # [[], ['W', 'A'], ['K', 'T']]
    # [['S', 'T'], ['E', 'Y'], ['T', 'K']]
    # [['T', 'S'], ['N', 'K'], ['W', 'E']]

PDB_list = os.listdir('./format_structure_data')
PDB_pesudo_site_mapping = {}
for PDB_file in PDB_list:
    pesudo_site = get_PDB_residure_site(PDB_file = PDB_file)
    PDB_pesudo_site_mapping[PDB_file] = pesudo_site

with open('PDB_pesudo_site_mapping.json', 'w') as file:
    json.dump(PDB_pesudo_site_mapping, file)

# 从文件中读取映射
#with open('PDB_pesudo_site_mapping.json', 'r') as file:
#    loaded_mapping = json.load(file)

#print(loaded_mapping)
