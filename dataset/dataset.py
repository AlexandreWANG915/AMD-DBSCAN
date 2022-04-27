import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import math
from sklearn.cluster import KMeans
import random
import collections
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KDTree
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from typing import List
import math
import pandas as pd


class make:

    def __init__(self, samples=500, centers=2, center_box=[0, 1], radius=[1, 1], centers_point=None, ratio=False,
                 seed=None):

        self.n_samples = samples
        self.centers = centers
        self.center_box = center_box
        self.radius = radius
        self.centers_point = centers_point
        self.x = 0
        self.y = 0
        self.XY = []
        self.labels = []
        self.ratio = ratio
        if seed is not None:
            random.seed(seed)

        n = []
        if self.ratio is False:
            for i in range(self.centers):
                n.append(int(self.n_samples / self.centers))
        else:
            for i in range(self.centers):
                n.append(int(self.ratio[i] * self.n_samples))

        num = 0
        for i in range(self.centers):
            self.labels.append([num for i in range(n[i])])
            num += 1
        self.labels = np.array([n for a in self.labels for n in a])

        theta = []
        r = []

        if self.centers_point is None:
            self.centers_point = []
            for i in range(self.centers):
                self.centers_point.append([random.randint(self.center_box[0] * 1e2, self.center_box[1] * 1e2) / 1e2,
                                           random.randint(self.center_box[0] * 1e2, self.center_box[1] * 1e2) / 1e2])

        for i in range(self.centers):
            theta.append([random.randint(0, 62831) / 1e4 for j in range(n[i])])
            r.append([random.randint(0, self.radius[i] * 1e4) / 1e4 for j in range(n[i])])

        x = []
        y = []
        for i in range(self.centers):
            x.append((np.cos(theta[i]) * r[i] + self.centers_point[i][0]).tolist())
            y.append((np.sin(theta[i]) * r[i] + self.centers_point[i][1]).tolist())
        X = [n for a in x for n in a]
        Y = [n for a in y for n in a]

        for i in range(len(X)):
            self.XY.append([X[i], Y[i]])

    def get_xy(self):
        return np.array(self.XY)

    def get_labels(self, noise=None):
        if noise is None:
            return self.labels
        noise_label = np.array([-1 for i in range(noise)])
        self.labels = np.hstack((self.labels, noise_label))
        return self.labels

    def get_noise(self, num):
        noise = [random.randint(self.center_box[0] * 2 * 1e2, self.center_box[1] * 2 * 1e2) / 1e2 for j in
                 range(num * 2)]
        noise = np.array(noise).reshape(-1, 2)
        return noise



if __name__ == "__main__":
    a = make(samples=5000, centers=6, center_box=[-1000, 1000], radius=[500, 30, 500, 30, 150, 150],
                  centers_point=[[-1000, 1000], [-100, -50], [1000, -1000], [153, 489], [350, -350], [-654, -841]],
                  seed=15)

    data = a.get_xy()
    label = a.get_labels()
