from collections import defaultdict
from math import inf
from math import sqrt
import random
import csv
'''
Author: Yue Yu
Date: 9/29/2019
K-means implementation
'''

def point_avg(points):
    """
    Accepts a list of points, each with the same number of dimensions.
    (points can have more dimensions than 2)
    
    Returns a new point which is the center of all the points.
    """
    if points is None or len(points) == 0:
        raise Exception("No input array")
    byAxis = zip(*points)
    ret = [0] * len(byAxis)
    for i in len(byAxis):
        ret[i] = (sum(byAxis[i]) / len(points))
    return ret
    


def update_centers(data_set, assignments):
    """
    Accepts a dataset and a list of assignments; the indexes 
    of both lists correspond to each other.
    Compute the center for each of the assigned groups.
    Return `k` centers in a list
    """
    if data_set is None or len(data_set) == 0 or assignments is None or len(assignments) == 0:
        raise Exception("Missing input")
    if len(data_set) != len(assignments):
        raise Exception("Input length does not match")
    byCluster = defaultdict(list)
    for (centroid, point) in zip(assignments, data_set):
        byCluster[centroid].append(point)
    ret = []
    for key in byCluster.keys():
        ret.append(point_avg(byCluster[key]))
    return ret


def assign_points(data_points, centers):
    """
    """
    assignments = []
    for point in data_points:
        shortest = inf  # positive infinity
        shortest_index = 0
        for i in range(len(centers)):
            val = distance(point, centers[i])
            if val < shortest:
                shortest = val
                shortest_index = i
        assignments.append(shortest_index)
    return assignments


def distance(a, b):
    """
    Returns the Euclidean distance between a and b
    """
    if a is None or b is None:
        raise Exception("Missing input")
    if len(a) != len(b):
        raise Exception("Wrong Dimension")
    ret = 0
    for i in range(len(a)):
        ret += (a[i] - b[i]) ** 2
    return sqrt(ret)

def generate_k(data_set, k):
    """
    Given `data_set`, which is an array of arrays,
    return a random set of k points from the data_set
    """
    if data_set is None or len(data_set) == 0 or k is None:
        raise Exception("Missing input")
    if k <= 0 or k > len(data_set):
        raise Exception("K is invalid")
    return random.sample(data_set, k)

def get_list_from_dataset_file(dataset_file):
    if dataset_file is None:
        raise Exception("Missing input")
    ret = []
    with open(dataset_file) as data:
        dataset = csv.reader(data)
        for row in dataset:
            rowList = []
            for col in row:
                rowList.append(int(col))
            ret.append(rowList)
    return ret

def cost_function(clustering):
    if clustering is None or len(clustering) == 0:
        raise Exception("Missing input")
    ret = 0
    for (k, v) in clustering.items():
        centroid = point_avg(v)
        for point in v:
            ret += distance(centroid, point) ** 2
    return ret

def k_means(dataset_file, k):
    dataset = get_list_from_dataset_file(dataset_file)
    k_points = generate_k(dataset, k)
    assignments = assign_points(dataset, k_points)
    old_assignments = None
    while assignments != old_assignments:
        new_centers = update_centers(dataset, assignments)
        old_assignments = assignments
        assignments = assign_points(dataset, new_centers)
    clustering = defaultdict(list)
    for assignment, point in zip(assignments, dataset):
        clustering[assignment].append(point)
    return clustering
