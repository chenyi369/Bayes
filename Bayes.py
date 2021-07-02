import pandas as pd
import numpy as np

def read_xlsx(csv_path):
    data = pd.read_excel(csv_path)
    print(data)
    return data

def train_test_split(data, test_size=0.2, random_state=None):
    index = data.shape[0]
    # 设置随机种子，当随机种子非空时，将锁定随机数
    if random_state:
        np.random.seed(random_state)
        # 将样本集的索引值进行随机打乱
        # permutation随机生成0-len(data)随机序列
    shuffle_indexs = np.random.permutation(index)
    # 提取位于样本集中20%的那个索引值
    test_size = int(index * test_size)
    # 将随机打乱的20%的索引值赋值给测试索引
    test_indexs = shuffle_indexs[:test_size]
    # 将随机打乱的80%的索引值赋值给训练索引
    train_indexs = shuffle_indexs[test_size:]
    # 根据索引提取训练集和测试集
    train = data.iloc[train_indexs]
    test = data.iloc[test_indexs]
    return train, test

def calculate(data):
    y = data[data.columns[-1]]  # 依据公式求某列特征的熵 目标变量作为概率依据
    n = len(y)
    count_y = {}
    pi_y = {}
    for i, j in y.value_counts().items():
        count_y[i] = j
        pi_y[i] = j/n
    return count_y, pi_y

def testcalculate(train,test):  #test = ["青年","否","否","一般"]
    count_y, pi_y = calculate(train)
    pi_list = []
    ally = pd.unique(train.iloc[:, -1])
    for y in ally:
        pi_x = 1
        features = list(train.columns[:-1])
        for i in range(len(features)):
            df = train[train[features[i]] == test[i]]
            df = df[df.iloc[:, -1] == y]
            pi_x = pi_x * len(df) / count_y[y]
            # print(pi_x)  #y=1条件下x所有特征乘积
        pi = pi_y[y] * pi_x
        pi_list.append(pi)
    # print(ally)
    # print(pi_list)
    pi_list = np.array(pi_list)
    index = np.argsort(-pi_list)
    label = ally[index[0]]
    return label

def accuracy(train,test):
    correct = 0
    for i in range(len(test)):
        tiaojian = test.iloc[i, :-1].values
        label = testcalculate(train, tiaojian)
        if test.iloc[[i], -1].values == label:
            correct += 1
    accuracy = (correct / float(len(test))) * 100.0
    print("Accuracy:", accuracy, "%")
    return accuracy


if __name__ == '__main__':
    data = read_xlsx(r'D:\数据集\daikuan.xlsx')
    train,test = train_test_split(data)
    # calculate(data)
    # print(labels[1])
    # testcalculate(data, ["青年","否","否","一般"])
    accuracy(train, test)