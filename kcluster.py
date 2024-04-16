############################################################
# This code makes the elbow graphs used to pick epsilon for
# DBSCAN
############################################################
import distance as d
import random, csv, math
from heapq import heappop, heappush, heapify
import numpy as np
import matplotlib.pyplot as plt  # 原来没有这行

def read_data(filename):
    with open(filename, "r") as file:
        file_data = csv.reader(file)
        next(file_data)
        data = []

        for row in file_data:
            #x = row[1:len(row) - 1]
            x = row[1:len(row) - 2]
            data.append(x)
    
    return data

def distance_plot(filename, k, metric):
    data = read_data("D:/fanss/try/data.csv")

    pts = data # makes the double loop clearer

    distances = []
    for pt in pts:
        # for each point, find the distances to the k closest neighbors
        point = np.array(pt)
        x = []
        heapify(x)
        for row in data:
            # using a heap, we keep track of the k lowest distances
            dist = float(-1 * metric(point, row)) # because heapq makes a min heap
            if len(x) <= k:
                heappush(x, dist)
            elif dist > x[0]:
                heappop(x)
                heappush(x, dist)
        distances.append(heappop(x))


    y = [-1 * d for d in distances] # undo the * -1 from earlier
    y.sort(reverse=True)
    #y = [y] # not 100% sure why this was here
            # delete if this isn't working

    plt.scatter(range(len(y)), y)  # 绘制散点图
    plt.show()

#对于数据集中的每个数据点，计算该数据点与其k个最近邻点之间的距离，并将这些距离按照降序的方式存储在distances列表中。distances[0]代表数据集中的第一个数据点与其k个最近邻点之间的最大距离；
distance_plot("D:/fanss/try/data.csv",10,d.independent_distance)
