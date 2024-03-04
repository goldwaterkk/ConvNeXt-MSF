from Bio.SubsMat import MatrixInfo
import math
import pandas as pd
import json

PDB_site_path = './bio_data/PDB_pseudo_site_mapping.json'
AAindex_path = './bio_data/AAindex_data.csv'
Kmean_path = "./bio_data/Kmeans_HLA_data.csv"


def read_PDB_pseudo_site_mapping():
    # 获得根据结构获得的残基数据
    with open(PDB_site_path, 'r') as file:
        loaded_mapping = json.load(file)
    return loaded_mapping


def read_aaindex_value():
    # 获得AAindex的数据
    df = pd.read_csv(AAindex_path)
    index_dict = df.set_index('Description').to_dict(orient='index')
    return index_dict


def read_HLA_kmeans_data():
    return pd.read_csv(Kmean_path)


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

def cosine_rate(now_step, total_step, end_lr_rate):
    rate = ((1 + math.cos(now_step * math.pi / total_step)) / 2) * (1 - end_lr_rate) + end_lr_rate  # cosine
    return rate


def cosine_scheduler(initial_lr, epochs, steps, warmup_epochs=1, end_lr_rate=1e-7 , train_writer=None):
    """custom learning rate scheduler"""
    assert warmup_epochs < epochs
    warmup = np.linspace(start=1e-8, stop=initial_lr, num=warmup_epochs * steps)
    remainder_steps = (epochs - warmup_epochs) * steps
    cosine = initial_lr * np.array([cosine_rate(i, remainder_steps, end_lr_rate) for i in range(remainder_steps)])
    lr_list = np.concatenate([warmup, cosine])

    for i in range(len(lr_list)):
        new_lr = lr_list[i]
        if train_writer is not None:
            # writing lr into tensorboard
            with train_writer.as_default():
                tf.summary.scalar('learning rate', data=new_lr, step=i)
        yield new_lr

def get_predict(Pre_y):
    Y = tf.keras.layers.Softmax()(Pre_y)  # Trans the logist predict value to label
    return np.argmax(Y, axis=1)


def get_confusion_matrix(y_true, y_pred):
    """
    Calculates the confusion matrix from given labels and predictions.
    Expects tensors or numpy arrays of same shape.
    """
    TP, FP, TN, FN = 0, 0, 0, 0

    for i in range(y_true.shape[0]):
        if y_true[i] == 0 and y_pred[i] == 0:
            TN += 1
        elif y_true[i] == 0 and y_pred[i] == 1:
            FP += 1
        elif y_true[i] == 1 and y_pred[i] == 0:
            FN += 1
        elif y_true[i] == 1 and y_pred[i] == 1:
            TP += 1

    conf_matrix = [
        [TP, FP],
        [FN, TN]
    ]

    return conf_matrix


def get_accuracy(conf_matrix):
    """
    Calculates accuracy metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    return (TP + TN) / (TP + FP + FN + TN)


def get_precision(conf_matrix):
    """
    Calculates precision metric from the given confusion matrix.
    """
    TP, FP = conf_matrix[0][0], conf_matrix[0][1]

    if TP + FP > 0:
        return TP / (TP + FP)
    else:
        return 0


def get_recall(conf_matrix):
    """
    Calculates recall metric from the given confusion matrix.
    """
    TP, FN = conf_matrix[0][0], conf_matrix[1][0]

    if TP + FN > 0:
        return TP / (TP + FN)
    else:
        return 0


def get_f1score(conf_matrix):
    """
    Calculates f1-score metric from the given confusion matrix.
    """
    p = get_precision(conf_matrix)
    r = get_recall(conf_matrix)

    if p + r > 0:
        return 2 * p * r / (p + r)
    else:
        return 0


def get_mcc(conf_matrix):
    """
    Calculates Matthew's Correlation Coefficient metric from the given confusion matrix.
    """
    TP, FP, FN, TN = conf_matrix[0][0], conf_matrix[0][1], conf_matrix[1][0], conf_matrix[1][1]
    if TP + FP > 0 and TP + FN > 0 and TN + FP > 0 and TN + FN > 0:
        return (TP * TN - FP * FN) / math.sqrt((TP + FP) * (TP + FN) * (TN + FP) * (TN + FN))
    else:
        return 0


def evaluate(Y_pred, Y_real):
    conf_matrix = get_confusion_matrix(Y_real, Y_pred)
    precision = get_precision(conf_matrix)
    recall = get_recall(conf_matrix)
    fscore = get_f1score(conf_matrix)
    mcc = get_mcc(conf_matrix)
    val_acc = get_accuracy(conf_matrix)

    return precision, recall, fscore, mcc, val_acc
