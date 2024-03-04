import pandas as pd
import numpy as np
from tensorflow.keras.utils import to_categorical
from utils import *
from mhcflurry.allele_encoding import AlleleEncoding
from mhcflurry.amino_acid import BLOSUM62_MATRIX
from numpy.testing import assert_equal
from mhcflurry.encodable_sequences import EncodableSequences
from mhcflurry import amino_acid
from model import model1_MHCflurry_CNN, model2_NetMHCpan4_CNN, model3_MSF_CNN, model4_ANN40_CNN, model5_DeepHLA_CNN

import tensorflow as tf


def readSeqInfo(filename):
    '''
    读取伪序列的序列信息
    '''
    fr = open(filename)
    raw_datas = fr.readlines()

    seq_dict = {}

    for i in range(len(raw_datas)):

        if raw_datas[i].startswith('>'):
            label = raw_datas[i].split(' ')[0].replace('>HLA_', '')  # 第一行以空格分隔 第0列是HLA名字将>HLA_去掉，len(raw_datas)是文件总行数
            seq = raw_datas[i + 1].replace('\n', '')  # 第二行伪序列

            seq_dict[label] = seq

    return seq_dict


def readSeqInfo2(filename):
    '''
    读取伪序列的序列信息 MHCflurry, NetMHCpan4.0
    '''
    fr = open(filename)
    raw_datas = fr.readlines()

    seq_dict = {}

    for i in range(len(raw_datas)):
        temp = raw_datas[i].split(' ')
        seq_dict[temp[0]] = temp[1].strip()

    return seq_dict


'''
    encoding = AlleleEncoding(
        ["A*02:01"],
        {
            "A*02:01": "ACDD",
        }
    )
'''


def Concat_reshape(data1, data2):
    data3 = np.concatenate((data1, data2), axis=1)
    data3 = np.expand_dims(data3, axis=3)
    # data3 = np.concatenate((data3, data3, data3, data3), axis=3)
    return data3


def MHCflurry_matrix(filename, file_data):  # (36+15)*21*1
    seq_dict = readSeqInfo2(filename)
    df = pd.read_csv(file_data)
    encoding = AlleleEncoding(
        # alleles=df['allele'].to_list(),
        alleles=df['HLA'].str.replace(':', '').to_list(),
        allele_to_sequence=seq_dict,
        borrow_from=None
    )
    encoder_alle = encoding.fixed_length_vector_encoded_sequences("BLOSUM62")

    encoded_peptides = EncodableSequences.create(df['peptide'].to_list())
    encoder_peptides = encoded_peptides.variable_length_to_fixed_length_vector_encoding("BLOSUM62")

    X = Concat_reshape(encoder_peptides, encoder_alle)
    Y = np.array(df['label'].to_list())
    return X, Y


def MSF_matrix(i):  # 9*6*5
    tt = np.load("./data/npy/k_folds/ms_fold_{}.npz".format(i))
    return tt['x'], tt['y']


def NetMHCpan4_matrix(filename, file_data):
    df = pd.read_csv(file_data)
    seq_dict = readSeqInfo2(filename)
    index_encoded_matrix = amino_acid.index_encoding(df['9mer'].to_list(), amino_acid.AMINO_ACID_INDEX)
    encoder_peptide = amino_acid.fixed_vectors_encoding(index_encoded_matrix,
                                                        amino_acid.ENCODING_DATA_FRAMES["BLOSUM62"])

    alle = []
    temp = df['HLA'].str.replace(':', '').to_list()
    for i in range(len(temp)):
        alle.append(seq_dict[temp[i]])

    index_encoded_matrix = amino_acid.index_encoding(alle, amino_acid.AMINO_ACID_INDEX)
    encoder_alle = amino_acid.fixed_vectors_encoding(index_encoded_matrix, amino_acid.ENCODING_DATA_FRAMES["BLOSUM62"])
    # (90803, 34, 21, 1)
    X = Concat_reshape(encoder_peptide, encoder_alle)
    Y = np.array(df['label'].to_list())
    return X, Y


def DeepHLA_transform(HLA, peptide):
    aa_idx = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
              'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21}
    data = HLA + peptide
    seq = data + 'X' * (49 - len(data))
    seq = [aa_idx[x] - 1 for x in seq]  # one-hot need -1
    return seq


def DeepHLA_matrix(filename, file_data):  # (90901, 49, 21, 1)
    df = pd.read_csv(file_data)
    HLA_seq = pd.read_csv(filename, sep='\t')
    seqs = {}
    for i in range(len(HLA_seq)):
        seqs[HLA_seq.HLA[i]] = HLA_seq.sequence[i]
    df['cost_cents'] = df.apply(
        lambda row: DeepHLA_transform(
            HLA=seqs[row['HLA'].replace(':', '')],
            peptide=row['peptide']),
        axis=1)
    X = np.vstack(df.cost_cents)
    X_one_hot = to_categorical(X, num_classes=21)
    X_one_hot = np.expand_dims(X_one_hot, axis=3)
    Y = np.array(df['label'].to_list())
    # X = np.concatenate((X_one_hot, X_one_hot, X_one_hot), axis=3) #无需扩展数据量
    return X_one_hot, Y


def ANN40_matrix(filename, file_data):  # (90901, 43, 21, 4)
    X_list = []
    aa_idx = {'A': 1, 'C': 2, 'D': 3, 'E': 4, 'F': 5, 'G': 6, 'H': 7, 'I': 8, 'K': 9, 'L': 10, 'M': 11, 'N': 12,
              'P': 13, 'Q': 14, 'R': 15, 'S': 16, 'T': 17, 'V': 18, 'W': 19, 'Y': 20, 'X': 21}
    df = pd.read_csv(file_data)
    seq_dict = readSeqInfo2(filename)
    # 将字符串通过aaidx转化为数字
    for peptide in df['9mer'].to_list():
        numbers = [aa_idx[char] for char in peptide]
        # 将数字转化为one-hot编码
        one_hot = np.zeros((len(numbers), len(aa_idx)))
        for i, num in enumerate(numbers):
            one_hot[i, num - 1] = 0.9  # 将1变为0.9
            one_hot[i, np.arange(len(aa_idx)) != num - 1] = 0.05  # 将0变为0.05
        X_list.append(one_hot)

    alle = []
    temp = df['HLA'].str.replace(':', '').to_list()
    for i in range(len(temp)):
        alle.append(seq_dict[temp[i]])

    index_encoded_matrix = amino_acid.index_encoding(alle, amino_acid.AMINO_ACID_INDEX)
    encoder_alle = amino_acid.fixed_vectors_encoding(index_encoded_matrix, amino_acid.ENCODING_DATA_FRAMES["BLOSUM62"])
    # (90803, 34, 21)
    X = Concat_reshape(X_list, encoder_alle)
    Y = np.array(df['label'].to_list())
    return X, Y


def train(index, times, load_weight=False):
    FILE_PATH = "./data/csv/k_folds/ms_fold_{}.csv"
    file_path_alle_NetMHCpan = "./data/pseudo/NetMHCpan_pseudo"
    file_path_alle_flurry = "./data/pseudo/MHCflurry_pseudosequences"
    file_path_alle_DeepHLA = "./data/pseudo/DeepHLA_MHC_pseudo.dat"

    # create models
    MHCflurry_CNN = model1_MHCflurry_CNN(2)
    NetMHCpan4_CNN = model2_NetMHCpan4_CNN(2)
    MSF_CNN = model3_MSF_CNN(2)
    ANN40_CNN = model4_ANN40_CNN(2)
    DeepHLA_CNN = model5_DeepHLA_CNN(2)

    if load_weight:  # 如果进行多次训练，则load_weight = True,表明获取上次的参数进行再次训练
        epoch = 7
        MHCflurry_CNN.load_weights(
            "./model/input_matrix_compare_5folds/MHCflurry_CNN_{}_{}.h5".format(epoch, times))
        NetMHCpan4_CNN.load_weights(
            "./model/input_matrix_compare_5folds/NetMHCpan4_CNN_{}_{}.h5".format(epoch, times))
        MSF_CNN.load_weights(
            "./model/input_matrix_compare_5folds/MSF_CNN_{}_{}.h5".format(epoch, times))
        ANN40_CNN.load_weights(
            "./model/input_matrix_compare_5folds/ANN40_CNN_{}_{}.h5".format(epoch, times))
        DeepHLA_CNN.load_weights(
            "./model/input_matrix_compare_5folds/DeepHLA_CNN_{}_{}.h5".format(epoch, times))

    epochs = 13
    for epoch in range(8, epochs):  # 训练轮次
        for i in index:  # 从index获取训练数据
            print("EPOCH:{}, MS_index:{}-----------".format(epoch, i))
            file_data = FILE_PATH.format(i)  # loads ms_fold_i.csv

            X, Y = MSF_matrix(i)
            Y = tf.one_hot(Y, 2)
            MSF_CNN.fit(X, Y)

            X, Y = MHCflurry_matrix(file_path_alle_flurry, file_data)
            Y = tf.one_hot(Y, 2)
            MHCflurry_CNN.fit(X, Y)

            X, Y = NetMHCpan4_matrix(file_path_alle_NetMHCpan, file_data)
            Y = tf.one_hot(Y, 2)
            NetMHCpan4_CNN.fit(X, Y)

            X, Y = ANN40_matrix(file_path_alle_NetMHCpan, file_data)
            Y = tf.one_hot(Y, 2)
            ANN40_CNN.fit(X, Y)

            X, Y = DeepHLA_matrix(file_path_alle_DeepHLA, file_data)
            Y = tf.one_hot(Y, 2)
            DeepHLA_CNN.fit(X, Y)

        MHCflurry_CNN.save_weights(
            "./model/input_matrix_compare_5folds/MHCflurry_CNN_{}_{}.h5".format(epoch, times))
        NetMHCpan4_CNN.save_weights(
            "./model/input_matrix_compare_5folds/NetMHCpan4_CNN_{}_{}.h5".format(epoch, times))
        MSF_CNN.save_weights(
            "./model/input_matrix_compare_5folds/MSF_CNN_{}_{}.h5".format(epoch, times))
        ANN40_CNN.save_weights(
            "./model/input_matrix_compare_5folds/ANN40_CNN_{}_{}.h5".format(epoch, times))
        DeepHLA_CNN.save_weights(
            "./model/input_matrix_compare_5folds/DeepHLA_CNN_{}_{}.h5".format(epoch, times))


def valid2(epoch, times, index):
    f = open("./logs/log_input_matrix_compare_5folds_{}.csv".format(index[0]), "w")
    f.write("times,precision, recall, fscore, mcc, val_acc,\n")
    f.close()

    FILE_PATH = "./data/csv/k_folds/ms_fold_{}.csv"

    f = open("./logs/log_input_matrix_compare_5folds_{}.csv".format(index[0]), "a")
    file_path_alle_DeepHLA = "./data/pseudo/DeepHLA_MHC_pseudo.dat"
    file_path_alle_NetMHCpan = "./data/pseudo/NetMHCpan_pseudo"
    file_path_alle_flurry = "./data/pseudo/MHCflurry_pseudosequences"

    # create models
    MHCflurry_CNN = model1_MHCflurry_CNN(2)
    NetMHCpan4_CNN = model2_NetMHCpan4_CNN(2)
    MSF_CNN = model3_MSF_CNN(2)
    ANN40_CNN = model4_ANN40_CNN(2)
    DeepHLA_CNN = model5_DeepHLA_CNN(2)

    MHCflurry_CNN.load_weights(
        "./model/input_matrix_compare_5folds/MHCflurry_CNN_{}_{}.h5".format(epoch, times))
    NetMHCpan4_CNN.load_weights(
        "./model/input_matrix_compare_5folds/NetMHCpan4_CNN_{}_{}.h5".format(epoch, times))
    MSF_CNN.load_weights(
        "./model/input_matrix_compare_5folds/MSF_CNN_{}_{}.h5".format(epoch, times))
    ANN40_CNN.load_weights(
        "./model/input_matrix_compare_5folds/ANN40_CNN_{}_{}.h5".format(epoch, times))
    DeepHLA_CNN.load_weights(
        "./model/input_matrix_compare_5folds/DeepHLA_CNN_{}_{}.h5".format(epoch, times))

    for i in index:
        print("---------{}--------".format(i))
        file_data = FILE_PATH.format(i)

        X, Y = MSF_matrix(i)
        temp_Y = MSF_CNN.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("MSF_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

        X, Y = MHCflurry_matrix(file_path_alle_flurry, file_data)
        temp_Y = MHCflurry_CNN.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("MHCflurry_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

        X, Y = ANN40_matrix(file_path_alle_NetMHCpan, file_data)
        temp_Y = ANN40_CNN.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("ANN40_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

        X, Y = DeepHLA_matrix(file_path_alle_DeepHLA, file_data)
        temp_Y = DeepHLA_CNN.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("DeepHLA_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

        X, Y = NetMHCpan4_matrix(file_path_alle_NetMHCpan, file_data)
        temp_Y = NetMHCpan4_CNN.predict(X)
        temp_Y = tf.argmax(temp_Y, axis=1)
        precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
        f.write("NetMHCpan4_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))
        f.write("\n")
    f.close()


def valid(epoch, times, index): # 将每个folds中的4个小folds进行合并统一测试
    f = open("./logs/log_input_matrix_compare_5folds_{}.csv".format(index[0]//4), "w")
    f.write("times,precision, recall, fscore, mcc, val_acc,\n")
    f.close()

    FILE_PATH = "./data/csv/k_folds/ms_fold_{}.csv"

    f = open("./logs/log_input_matrix_compare_5folds_{}.csv".format(index[0]//4), "a")
    file_path_alle_DeepHLA = "./data/pseudo/DeepHLA_MHC_pseudo.dat"
    file_path_alle_NetMHCpan = "./data/pseudo/NetMHCpan_pseudo"
    file_path_alle_flurry = "./data/pseudo/MHCflurry_pseudosequences"

    # create models
    MHCflurry_CNN = model1_MHCflurry_CNN(2)
    NetMHCpan4_CNN = model2_NetMHCpan4_CNN(2)
    MSF_CNN = model3_MSF_CNN(2)
    ANN40_CNN = model4_ANN40_CNN(2)
    DeepHLA_CNN = model5_DeepHLA_CNN(2)

    MHCflurry_CNN.load_weights(
        "./model/input_matrix_compare_5folds/MHCflurry_CNN_{}_{}.h5".format(epoch, times))
    NetMHCpan4_CNN.load_weights(
        "./model/input_matrix_compare_5folds/NetMHCpan4_CNN_{}_{}.h5".format(epoch, times))
    MSF_CNN.load_weights(
        "./model/input_matrix_compare_5folds/MSF_CNN_{}_{}.h5".format(epoch, times))
    ANN40_CNN.load_weights(
        "./model/input_matrix_compare_5folds/ANN40_CNN_{}_{}.h5".format(epoch, times))
    DeepHLA_CNN.load_weights(
        "./model/input_matrix_compare_5folds/DeepHLA_CNN_{}_{}.h5".format(epoch, times))

    all_X_MSF = []
    all_Y_MSF = []

    all_X_MHCflurry = []
    all_Y_MHCflurry = []

    all_X_ANN = []
    all_Y_ANN = []

    all_X_NetMHC = []
    all_Y_NetMHC = []

    all_X_DeepHLA = []
    all_Y_DeepHLA = []
    for i in index:
        file_data = FILE_PATH.format(i)
        X_MSF, Y_MSF = MSF_matrix(i)
        all_X_MSF.append(X_MSF)
        all_Y_MSF.append(Y_MSF)

        X_MHCflurry, Y_MHCflurry = MHCflurry_matrix(file_path_alle_flurry, file_data)
        all_X_MHCflurry.append(X_MHCflurry)
        all_Y_MHCflurry.append(Y_MHCflurry)

        X_ANN, Y_ANN = ANN40_matrix(file_path_alle_NetMHCpan, file_data)
        all_X_ANN.append(X_ANN)
        all_Y_ANN.append(Y_ANN)

        X_DeepHLA, Y_DeepHLA = DeepHLA_matrix(file_path_alle_DeepHLA, file_data)
        all_X_DeepHLA.append(X_DeepHLA)
        all_Y_DeepHLA.append(Y_DeepHLA)

        X_NetMHC, Y_NetMHC = NetMHCpan4_matrix(file_path_alle_NetMHCpan, file_data)
        all_X_NetMHC.append(X_NetMHC)
        all_Y_NetMHC.append(Y_NetMHC)


    X = np.concatenate(all_X_MSF, axis=0)
    Y = np.concatenate(all_Y_MSF, axis=0)
    temp_Y = MSF_CNN.predict(X)
    temp_Y = tf.argmax(temp_Y, axis=1)
    precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
    f.write("MSF_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

    X = np.concatenate(all_X_MHCflurry, axis=0)
    Y = np.concatenate(all_Y_MHCflurry, axis=0)
    temp_Y = MHCflurry_CNN.predict(X)
    temp_Y = tf.argmax(temp_Y, axis=1)
    precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
    f.write("MHCflurry_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

    X = np.concatenate(all_X_ANN, axis=0)
    Y = np.concatenate(all_Y_ANN, axis=0)
    temp_Y = ANN40_CNN.predict(X)
    temp_Y = tf.argmax(temp_Y, axis=1)
    precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
    f.write("ANN40_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

    X = np.concatenate(all_X_DeepHLA, axis=0)
    Y = np.concatenate(all_Y_DeepHLA, axis=0)
    temp_Y = DeepHLA_CNN.predict(X)
    temp_Y = tf.argmax(temp_Y, axis=1)
    precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
    f.write("DeepHLA_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))

    X = np.concatenate(all_X_NetMHC, axis=0)
    Y = np.concatenate(all_Y_NetMHC, axis=0)
    temp_Y = NetMHCpan4_CNN.predict(X)
    temp_Y = tf.argmax(temp_Y, axis=1)
    precision, recall, fscore, mcc, val_acc = evaluate(np.array(temp_Y), Y)
    f.write("NetMHCpan4_CNN_{}, {}, {}, {}, {}, {}\n".format(times, precision, recall, fscore, mcc, val_acc))
    f.write("\n")
    f.close()

def main_5folds():
    mapping = {}
    for i in range(20):  # 将20个csv文件进行合并，合并为5个folds，[0,4] [5,8] ...
        if (i // 4 in mapping):
            mapping[i // 4].append(i)
        else:
            mapping[i // 4] = [i]

    print(mapping)

    for i in range(5):  # 5folds, times = 0, valid:[0,4], train:others
        train_index = []
        valid_index = []
        for j in range(5):
            if i != j:
                train_index += mapping[j]
            else:
                valid_index = mapping[j]
        train(train_index, times=i, load_weight=True)
        # valid(valid_index, times=i)


def train_cnn_ms():
    model_conext = model3_DE_CNN(2)
    for j in range(3):
        for i in range(20):
            temp = np.load("./data/npy/ms_train/train_{}.npz".format(i))
            Y = tf.one_hot(temp['y'], 2)
            model_conext.fit(temp['x'], Y)
            del temp
        model_conext.save_weights("./save_model_weight/ms_cnn/cnn_{}.h5".format(j))


def train_cnn_af():
    de_cnn_af = DE_CNN_AF(2)
    for j in range(10):
        for i in range(5):
            Y = []
            temp = np.load("./data/npy/af_train/af_{}.npz".format(i))
            for k in temp['y']:
                Y.append([1 - k, k])

            de_cnn_af.fit(temp['x'], np.array(Y), batch_size=32)
            del temp
        de_cnn_af.save_weights("./save_model_weight/DE_CNN_AF/cnn_{}.h5".format(j))


def valid_af_de_CNN():
    model_cnn = DE_CNN_AF(2)
    X = np.load("./data/npy/af_valid/af_valid.npz")['x']
    df = pd.read_csv("./data/csv/af_valid_data.csv")
    Y_ture = df['log50k'].to_list()
    for i in range(10):
        ans = []
        model_cnn.load_weights("./save_model_weight/DE_CNN_AF/cnn_{}.h5".format(i))
        Y_pre = model_cnn.predict(X)
        f = open("Y_pre.txt", 'w')
        for j in range(len(Y_ture)):
            ans.append(math.pow(Y_pre[j][1] - Y_ture[j], 2))
            f.write("{}\n".format(Y_pre[j][1]))
        f.close()
        print("{}  mean:{}  std:{}".format(i, np.mean(ans), np.std(ans)))


def main_valid():
    mapping = {}
    for i in range(20):  # 将20个csv文件进行合并，合并为5个folds，[0,4] [5,8] ...
        if (i // 4 in mapping):
            mapping[i // 4].append(i)
        else:
            mapping[i // 4] = [i]

    for i in range(5):  # 5folds, times = 0, valid:[0,4], train:others
        train_index = []
        valid_index = []
        for j in range(5):
            if i != j:
                train_index += mapping[j]
            else:
                valid_index = mapping[j]
        valid(epoch=12, times=i, index=valid_index)


main_valid()
# main_5folds()
