import collections
import time
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math
from sklearn.cluster import KMeans
import test
import parameterAdaptive
from sklearn import metrics
import pandas as pd

class dataset:

    def __init__(self, dpID=-1):
        """
        :param dpID:数据的编号
        """
        # dpID: 设置数据的编号
        # dimension: 数据的坐标
        # clusterId: 数据的类别
        # visited: 数据是否访问过
        self.dpID = dpID
        self.dimension = []
        self.clusterId = -1
        self.visited = None
        self.round = 0

    def SetDimension(self, dimension):
        for i in range(len(dimension)):
            self.dimension.append(dimension[i])

    def GetDimension(self):
        return self.dimension

    def GetDpId(self):
        return self.dpID

    def SetDpId(self, dpID):
        self.dpID = dpID

    def isVisited(self):
        return self.visited

    def SetVisited(self, visited):
        self.visited = visited

    def GetClusterId(self):
        return self.clusterId

    def SetClusterId(self, clusterId):
        self.clusterId = clusterId

    def GetRound(self):
        return self.round

    def SetRound(self, round):
        self.round = round

class kmeans_dis:

    def __init__(self, data, k=4, knn=1):
        """

        :param data: 数据集(二维列表)
        :param k: 建立kdis图所用的k
        """
        self.data = []
        self.k = k
        self.eps = []
        self.kdis = []
        self.minpts = []
        self.labels = []

        # 将数据集全部用dataset定义并保存起来
        for i in range(len(data)):
            tempdata = dataset()
            tempdata.SetDimension(data[i])
            tempdata.SetDpId(i)
            tempdata.SetVisited(False)
            tempdata.SetClusterId(-1)
            self.data.append(tempdata)

        self._get_kdis(data)
        self._get_eps(knn)
        x = [x + 1 for x in range(len(self.kdis))]

    def get_eps(self):
        """

        :return: 计算得到的密度半径
        """
        return self.eps

    def get_minpts(self):
        """

        :return: 计算得到的密度半径
        """
        return self.minpts

    def fit(self):
        data = self._get_data([x for x in range(len(self.data))])
        round = 0   # 表示层次聚类的轮次数
        labels_index = []

        # 遍历每个半径
        for k in range(len(self.eps)):
            if data == []:
                break
            radius = self.eps[k]
            minpts = self._get_minpts(data, self.eps[k])
            self.minpts.append(minpts)
            if minpts < 2:
                continue
            count = 0  # 计数器
            round += 1
            objs = DBSCAN(eps=radius, min_samples=minpts).fit_predict(data)
            # 记录聚类结果的最大编号i,则下一轮聚类的编号从i+1开始
            labels_index.append(max(objs))
            # 保存下一轮要聚类的数据
            data = []

            # 遍历整个数据集将聚类结果保存到对应数据信息中
            for i in range(len(self.data)):
                if not self.data[i].isVisited():
                    self.data[i].SetVisited(True)
                    self.data[i].SetClusterId(objs[count])
                    self.data[i].SetRound(round)
                    count += 1

            # 将下一轮要聚类的数据保存到data
            for i in range(len(self.data)):
                if self.data[i].GetClusterId() == -1:
                    data.append(self.data[i].GetDimension())
                    self.data[i].SetVisited(False)

        # 重新分配聚类结果
        plus = 0
        for i in range(round):
            if labels_index[i] == -1:
                continue
            plus += labels_index[i] + 1
            for j in range(len(self.data)):
                if self.data[j].GetRound() == i + 2 and self.data[j].GetClusterId() != -1:
                    self.data[j].SetClusterId(self.data[j].GetClusterId() + plus)

    def get_cluster_id(self):
        """

        :return: 返回所有数据的聚类结果
        """
        labels = []
        for i in range(len(self.data)):
            labels.append(self.data[i].GetClusterId())
        return labels

    def _get_data(self, index):
        """

        :param index: 保存dataset类型列表的项数
        :return: 数据的坐标信息，二维列表保存
        """
        data = []
        for i in index:
            data.append(self.data[i].GetDimension())
        return data

    def _get_kdis(self, data):
        """

        :param data: 输入数据，二维列表保存
        :return: 建立kdis图
        """
        tree = KDTree(data)
        dist, idx = tree.query(data, k=self.k+1)

        dist = np.transpose(dist).tolist()
        self.kdis = dist[-1]
        self.kdis = sorted(self.kdis)
        x = [x + 1 for x in range(len(self.kdis))]

        count = []
        cache = 0
        for i in self.kdis:
            if cache == i:
                continue
            cache = i
            count.append(self.kdis.count(i))

    def _get_slope(self, x, y) -> float:
        return (self.kdis[y] - self.kdis[x]) / (y - x)

    def _get_eps(self, knn):
        kmeans = KMeans(n_clusters=knn)
        kdis = np.array(self.kdis).reshape(len(self.kdis),1).tolist()
        label = kmeans.fit_predict(kdis)
        eps = kmeans.cluster_centers_
        self.eps = np.array(eps).reshape(len(eps)).tolist()
        self.eps = sorted(self.eps)

    def _get_minpts(self, data, eps) -> int:
        tree = KDTree(data)
        index = tree.query_radius(data, r=eps)
        count = 0
        for j in range(len(index)):
            count += len(index[j])
        minpts = (count / len(self.data))-1
        return math.ceil(minpts)

if __name__ == "__main__":
    data, label = datasets.make_blobs(n_samples=5000, n_features=2, centers=3,
                                       center_box=[-100, 100], random_state=2)
    epss, minptss = parameterAdaptive.paramterAdaptive(data)
    cluster = kmeans_dis(data, int(minptss), 3)
    eps = cluster.get_eps()
    print(eps)
    cluster.fit()
    y_pred = cluster.get_cluster_id()
    print(y_pred)




