import numpy as np
import math as m
import random
import matplotlib.pyplot as plt
import evaluate as eva
import pickle

# flame.txt
# Jain_cluster=2.txt
# Aggregation_cluster=7.txt
# Spiral_cluster=3.txt
# Pathbased_cluster=3.txt

data_path = "data/Aggregation_cluster=7.txt"


def save_obj(obj, name):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open(name + '.pkl', 'rb') as f:
        return pickle.load(f)

def load_my_data():
    data = np.load('con_out_all.npz', allow_pickle=True)
    return data['outputs'][2]


# 划分数据集
def split_my_data():
    npz_data = np.load('con_out_all_new.npz', allow_pickle=True)

    data = npz_data['outputs']
    labels = npz_data['labels']
    names = npz_data['names']

    # { '0': {'0':   {'data':[], 'label':[], 'name':[]}   } }

    dict_data = {'0': {}, '1': {}, '2': {}, '3': {}, '4': {}, '5': {}, '6': {}, '7': {}}
    count = [2, 2, 2, 2, 2, 4, 6, 20]

    for i in range(8):
        for j in range(count[i]):
            dict_data[str(i)][str(j)] = {}
            dict_data[str(i)][str(j)]['data'] = []
            dict_data[str(i)][str(j)]['label'] = []
            dict_data[str(i)][str(j)]['name'] = []

    for index in range(8):
        # 划分属性
        attribute = np.array(labels)[:, index: index + 1]
        for item_image in range(len(attribute)):
            dict_data[str(index)][str(attribute[item_image][0])]['data'].append(data[item_image])
            dict_data[str(index)][str(attribute[item_image][0])]['label'].append(labels[item_image])
            dict_data[str(index)][str(attribute[item_image][0])]['name'].append(names[item_image])

    return dict_data



# 导入数据
def load_data():
    points = np.loadtxt(data_path, delimiter=',')
    return points


def cal_dis(data, clu, k):
    """
    计算质点与数据点的距离
    :param data: 样本点
    :param clu:  质点集合
    :param k: 类别个数
    :return: 质心与样本点距离矩阵
    """
    dis = []
    for i in range(len(data)):
        dis.append([])
        for j in range(k):
            temp_value = 0
            for ii in range(len(data[i])):
                temp_value += (data[i, ii] - clu[j, ii])**2
            dis[i].append(m.sqrt(temp_value))
    return np.asarray(dis)


def divide(data, dis):
    """
    对数据点分组
    :param data: 样本集合
    :param dis: 质心与所有样本的距离
    :param k: 类别个数
    :return: 分割后样本
    """
    clusterRes = [0] * len(data)
    for i in range(len(data)):
        seq = np.argsort(dis[i])
        clusterRes[i] = seq[0]

    return np.asarray(clusterRes)


def center(data, clusterRes, k):
    """
    计算质心
    :param group: 分组后样本
    :param k: 类别个数
    :return: 计算得到的质心
    """
    clunew = []
    for i in range(k):
        # 计算每个组的新质心
        idx = np.where(clusterRes == i)
        sum = data[idx].sum(axis=0)
        avg_sum = sum/len(data[idx])
        clunew.append(avg_sum)
    clunew = np.asarray(clunew)
    return clunew[:, :]


def classfy(data, clu, k):
    """
    迭代收敛更新质心
    :param data: 样本集合
    :param clu: 质心集合
    :param k: 类别个数
    :return: 误差， 新质心
    """
    clulist = cal_dis(data, clu, k)
    clusterRes = divide(data, clulist)
    clunew = center(data, clusterRes, k)
    err = clunew - clu
    return err, clunew, k, clusterRes


def plotRes(data, clusterRes, clusterNum):
    """
    结果可视化
    :param data:样本集
    :param clusterRes:聚类结果
    :param clusterNum: 类个数
    :return:
    """
    nPoints = len(data)
    scatterColors = ['black', 'blue', 'green', 'yellow', 'red', 'purple', 'orange', 'brown']
    for i in range(clusterNum):
        color = scatterColors[i % len(scatterColors)]
        x1 = [];  y1 = []
        for j in range(nPoints):
            if clusterRes[j] == i:
                x1.append(data[j, 0])
                y1.append(data[j, 1])
        plt.scatter(x1, y1, c=color, alpha=1, marker='+')
    plt.show()

def select_representative(clulist, data, k):
    representative_indexs = []
    for ii in range(k):
        temp = clulist[:, ii:ii+1].tolist()  # 按照聚类中心取出
        representative_index = temp.index(min(temp))
        print("代表索引", representative_index)
        representative_indexs.append(representative_index)
    return representative_indexs


if __name__ == '__main__':
    k = 7  # 类别个数
    count = [2, 2, 2, 2, 2, 4, 6, 20]
    represent_index = []
    represent_label = {}
    represent_name = {}

    for i in range(8):
        for j in range(count[i]):
            dic_split = split_my_data()
            data = np.array(dic_split[str(i)][str(j)]['data'])
            labels = np.array(dic_split[str(i)][str(j)]['label'])
            names = np.array(dic_split[str(i)][str(j)]['name'])
            print("shape-->", data.shape)

            clu = random.sample(data[:, :].tolist(), k)  # 随机取质心
            clu = np.asarray(clu)
            err, clunew, k, clusterRes = classfy(data, clu, k)
            while np.any(abs(err) > 0):
                err, clunew, k, clusterRes = classfy(data, clunew, k)

            # clulist 数据存放格式 n*k
            clulist = cal_dis(data, clunew, k)  # 计算聚类中心与所有样本点的距离
            clusterResult = divide(data, clulist)  # 样本点划分

            select_index = select_representative(clulist, data, k)
            represent_label[str(i) + '-' + str(j)] = []
            represent_name[str(i) + '-' + str(j)] = []
            for select_index_item in select_index:
                represent_label[str(i) + '-' + str(j)].append(labels[select_index_item])
                represent_name[str(i) + '-' + str(j)].append(names[select_index_item])

            represent_index.append(select_index)

            nmi, acc, purity = eva.eva(clusterResult, np.asarray(data[:, 2]))
            plotRes(data, clusterResult, k)
            print("聚类中心", clunew)

    np.savez_compressed('index.npz', index=represent_index)

    save_obj(represent_label, 'represent_label')
    save_obj(represent_name, 'represent_name')





    # data = np.array(split_my_data()['0']['0'])
    # print("shape-->", data.shape)
    #
    # clu = random.sample(data[:, 0:-1].tolist(), k)  # 随机取质心
    # clu = np.asarray(clu)
    # err, clunew,  k, clusterRes = classfy(data, clu, k)
    # while np.any(abs(err) > 0):
    #     err, clunew,  k, clusterRes = classfy(data, clunew, k)
    #
    # clulist = cal_dis(data, clunew, k)
    # clusterResult = divide(data, clulist)
    #
    # nmi, acc, purity = eva.eva(clusterResult, np.asarray(data[:, 2]))
    # plotRes(data, clusterResult, k)
    # print("聚类中心", clunew)