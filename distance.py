############################################################
# This is code for the distance metric/semi-metric. Many 
# things, including the name of the datafile but also some
# details like how many empty rows the file has, are hard-
# coded and may thus require some effor to adapt. If you are
# not using the probability-based semimetric from the paper
# you can probably ignore this.
############################################################

import csv, json, copy, math
import numpy as np
from sklearn.metrics import pairwise_distances

probabilities = None
data = None

#跳过了第一列id和最后一列wscore
def read_data(filename):
    with open(filename, "r") as file:
        file_data = csv.reader(file)
        row_length = len(next(file_data))
        file_length = sum(1 for row in file_data)
        #data = np.empty((file_length, row_length - 2))
        data = np.empty((file_length, row_length - 3))   #data最后一列加上HH6 area

        file.seek(0)
        file_data = csv.reader(file)
        count = 0
        for row in file_data:
            if count > 0:
                #x = np.array(row[1:len(row) - 1])
                x = np.array(row[1:len(row) - 2])
                for i in range(len(x)):
                    data[count - 1][i] = x[i]
            count += 1
    return data


def precompute_overall_distances():
    global data
    if data is None:
        data = read_data("D:/fanss/try/data.csv")

    distances = np.empty((len(data), len(data)))
    for i in range(len(data)):
        for j in range(len(data)):
            print((i,j))
            if i == j:
                distances[i][j] = 0
            else:
                x = (data[i] == data[j])
                count = 0
                for k in range(len(data)):
                    y = (data[i] == data[k])
                    more_similar = True
                    for l in range(len(x)):
                        if x[l] and not y[l]:
                            more_similar = False

                    if more_similar:
                        count += 1
                distances[i][j] = count/len(data)


    # #with open("data/distances.json", "w") as file:
    # with open("D:/fanss/data/distances.json", "w") as file:
    #     json.dump(distances.tolist(), file)
    return distances


def precompute_individual_probabilities():
    global data
    if data is None:
        data = read_data("D:/fanss/try/data.csv")

    x = copy.deepcopy(data[0])
    for i in range(1,len(data)):
        x += data[i]
    print("probabilities computed")
    return x / len(data)


def independent_distance(x,y):  
    global probabilities, data

    if data is None:
        data = read_data("D:/fanss/try/data.csv")

        if probabilities is None:
            probabilities = precompute_individual_probabilities()

    b_array = (x == y)

    if b_array.all():
        return 0
    else:
        p = 1
        for i in range(len(b_array)):
            if b_array[i]:
                p *= ((probabilities[i]**2) + ((1-probabilities[i])**2))
        return p


def compute_distance_matrix():
    global data, probabilities
    if data is None:
        data = read_data("D:/fanss/try/data.csv")

    if probabilities is None:
        probabilities = precompute_individual_probabilities()

    k = pairwise_distances(data, metric=independent_distance)
    print("distance matrix computed")
    return k

# data = read_data("D:/fanss/try/data.csv")
# # print(data[0])

# distances = compute_distance_matrix()
# print("距离矩阵:", distances)