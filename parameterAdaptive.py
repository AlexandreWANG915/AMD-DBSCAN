from sklearn.metrics import euclidean_distances
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import time
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from evaluate import *
import pandas as pd


def getMinpts(matrix, eps):
    """
    :param matrix: 距离矩阵
    :param eps:
    :return: minpts
    """
    return np.round(np.sum(matrix < eps) / len(matrix))


def paramterAdaptive(data):
    """
    :param data: 待聚类数据
    :return: eps, minpts, K
    """
    # 数据长度
    N = len(data)
    # 计算升序排列的距离矩阵
    distances = np.sort(euclidean_distances(data), axis=1)
    # 距离矩阵求平均获得eps列表
    Deps = np.mean(distances[:, 1:N], axis=0)
    # 聚类获得簇的数目
    db = DBSCAN(eps=Deps[1], min_samples=getMinpts(distances, Deps[1])).fit(data)
    labels = db.labels_
    clusters = len(set(labels)) - (1 if -1 in labels else 0)
    # counter用来计算相同簇数出现的次数，如果大于3，认为聚类的簇数趋于稳定
    counter = 1
    for i in range(2, len(data) - 2):
        db = DBSCAN(eps=Deps[i], min_samples=getMinpts(distances, Deps[i])).fit(data)
        labels = db.labels_
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters_ == clusters:
            counter += 1
            if counter < BALANCENUM:
                continue
            else:
                # 当聚类获得簇的数目平稳下来时，认为此时为该数据最终的簇数
                best_clusters = n_clusters_
                # print("clusters=", best_clusters)
                left = i
                right = len(data) - 2
                n_clusters = 0
                mid = 0
                while right - left > 1:
                    mid = int(np.floor((left + right) / 2))
                    db = DBSCAN(eps=Deps[mid], min_samples=getMinpts(distances, Deps[mid])).fit(data)
                    labels = db.labels_
                    n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
                    if n_clusters < best_clusters:
                        right = mid
                    else:
                        left = mid
                if n_clusters == best_clusters:
                    print(mid)
                    return Deps[mid], getMinpts(distances, Deps[mid])
                else:
                    print(left)
                    return Deps[left], getMinpts(distances, Deps[left])
        else:
            clusters = n_clusters_
            counter = 1


def scoreAdaptive(data):
    N = len(data)
    distances = np.sort(euclidean_distances(data), axis=1)
    Deps = np.mean(distances[:, 1:N], axis=0)
    clusters = []
    # multual_info_score = []
    # v_measure_score = []
    silhouette_score = []
    for i in range(len(Deps) - 2):
        db = DBSCAN(eps=Deps[i], min_samples=getMinpts(distances, Deps[i])).fit(data)
        labels = db.labels_
        cluster = len(set(labels)) - (1 if -1 in labels else 0)
        if cluster == 1:
            break
        clusters.append(cluster)
        # multual_info_score.append(metrics.adjusted_mutual_info_score(labels_true, labels))
        # v_measure_score.append(metrics.v_measure_score(labels_true, labels))
        silhouette_score.append(metrics.silhouette_score(data, labels))
    # totalcore = np.array(multual_info_score) + np.array(v_measure_score)
    # index = np.argmax(totalcore)
    index = np.argmax(silhouette_score)
    print(index)
    x = range(len(clusters))
    plt.scatter(x, clusters)
    plt.xlabel("K")
    plt.ylabel("clusters numbers")
    plt.show()
    plt.scatter(Deps[:len(clusters)], clusters)
    plt.xlabel("eps")
    plt.ylabel("clusters numbers")
    plt.show()
    plt.scatter(x[10:], clusters[10:])
    plt.xlabel("K")
    plt.ylabel("clusters numbers")
    plt.show()
    # plt.scatter(x, multual_info_score)
    # plt.xlabel("K")
    # plt.ylabel("multual_info_score")
    # plt.show()
    # plt.scatter(x, v_measure_score)
    # plt.xlabel("K")
    # plt.ylabel("v_measure_score")
    # plt.show()
    plt.scatter(x, silhouette_score)
    plt.xlabel("K")
    plt.ylabel("silhouette_score")
    plt.show()
    # plt.scatter(x, totalcore)
    # plt.xlabel("K")
    # plt.ylabel("v_measure_score")
    # plt.show()
    return Deps[index], getMinpts(distances, Deps[index])


def makeGraph(data, labels_true):
    N = len(data)
    distances = np.sort(euclidean_distances(data), axis=1)
    Deps = np.mean(distances[:, 1:N], axis=0)
    clusters = []
    silhouette_score = []
    for i in range(1, len(Deps)):
        db = DBSCAN(eps=Deps[i], min_samples=getMinpts(distances, Deps[i])).fit(data)
        labels = db.labels_
        cluster = len(set(labels)) - (1 if -1 in labels else 0)
        # if cluster == 1:
        #     break
        clusters.append(cluster)
        silhouette_score.append(metrics.v_measure_score(labels_true, labels))
    print(clusters)
    x = range(len(clusters))
    plt.scatter(x, clusters,color="black", linewidth=1.0)
    plt.xlabel("K")
    plt.ylabel("clusters numbers")
    plt.show()
    # plt.scatter(Deps, clusters)
    # plt.xlabel("eps")
    # plt.ylabel("clusters numbers")
    # plt.show()
    plt.plot(x, clusters, linewidth=3.0)
    plt.annotate(r'best index',
                 xy=(330, clusters[330]), xycoords='data',
                 xytext=(+10, +30), textcoords='offset points', fontsize=12,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("index")
    plt.ylabel("clusters numbers")
    plt.savefig("clusternum.png", dpi=720)
    plt.show()

    plt.plot(x, silhouette_score, linewidth=3.0)
    plt.annotate(r'best index',
                 xy=(330, silhouette_score[330]), xycoords='data',
                 xytext=(30, -50), textcoords='offset points', fontsize=12,
                 arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=.2"))
    plt.xlabel("index")
    plt.ylabel("v_measure_score")
    plt.savefig("score.png", dpi=720)
    plt.show()

    plt.scatter(x, silhouette_score)
    plt.xlabel("index")
    plt.ylabel("silhouette_score")
    plt.show()


# flame.txt
# Jain_cluster=2.txt
# Aggregation_cluster=7.txt
# Spiral_cluster=3.txt
# Pathbased_cluster=3.txt
# data_path = "./data/unbalance.txt"


data_path = "./数据集/R15.txt"
BALANCENUM = 3


def load_data():
    """
    导入数据
    :return: 数据,标签
    """
    # data = np.loadtxt(data_path, delimiter=' ')
    data = np.loadtxt(data_path, delimiter=',')
    # 带标签
    return data[:, 0:2], data[:, 2]
    # 不带标签
    # return data


if __name__ == '__main__':
    data, labels_true = make_blobs(n_samples=1000, centers=3, cluster_std=0.5, random_state=0,
                                   center_box=(-20.0, 20.0))
    #
    # data, labels_true = load_data()
    # sheet = pd.read_excel('数据集.xlsx', header=None, sheet_name=2)
    # sheet = np.array(sheet)
    # data = sheet[:, :2]
    # label = np.array(sheet[:, 2], dtype=int)
    # data = load_data()
    # print(len(data))
    # data = StandardScaler().fit_transform(data)
    # makeGraph(data, labels_true)
    t = []
    num = 1   # 循环次数
    for i in range(num):
        t0 = time.time()
        eps, minpts = paramterAdaptive(data)
        t.append(time.time() - t0)
    t = np.array(t)
    tmax = min(t)
    taver = t.sum() / num
    print(eps, minpts)
    print("最短自适应时长：", tmax)
    print("平均自适应时长：", taver)
    y_pred = DBSCAN(eps=eps, min_samples=minpts).fit_predict(data)
    plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    plt.show()



    # from dbscan import *
    # dbscan = DBSCAN(data, label, k=minpts, eps=eps)
    # dbscan.fit()

    # print(label)
    # print(y_pred)

    # print((label == y_pred).sum() / len(label))
    # print(1- (y_pred<0).sum() / len(y_pred))
    # print("互信息：", metrics.adjusted_mutual_info_score(labels_true, y_pred))
    # print("v_measure：", metrics.v_measure_score(labels_true, y_pred))
    # print("轮廓系数：", metrics.silhouette_score(data, y_pred))
    # plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    # plt.show()

    # t1 = time.time()
    # # eps, minpts = scoreAdaptive(data, labels_true)
    # eps, minpts = scoreAdaptive(data, labels_true)
    # print("方法2自适应时长：", time.time() - t1)
    # y_pred = DBSCAN(eps=eps, min_samples=minpts).fit_predict(data)
    # plt.scatter(data[:, 0], data[:, 1], c=y_pred)
    # plt.show()
