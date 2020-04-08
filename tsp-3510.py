import sys
from pprint import pprint
import scipy.spatial.distance
import numpy as np
import time

np.set_printoptions(suppress=True)

# Needs improvement
# -numpy is converting ints to floats, not sure if it slows it down

# This function reads a given file and returns a dictionary
# with node number as keys and values as rounded coordinates
def read_coordinates(inputFile):
    nodeCoordMapping = {}
    rawText = open(inputFile, 'r').readlines()
    for line in rawText:
        rowList = line.split(' ')
        rowList[2] = rowList[2].replace('\n', '')
        rowList = [round(float(item)) for item in rowList]
        npArray = np.array([rowList[1], rowList[2]])
        nodeCoordMapping[rowList[0]] = npArray
    return nodeCoordMapping

def numpy_import(inputFile):
    nodeCoordMapping = {}
    npImport = np.loadtxt(inputFile, delimiter=' ', dtype=float)
    for array in npImport:
        nodeCoordMapping[array[0]] = np.rint(array[1:]).astype(int)
    return nodeCoordMapping

def create_graph(inputDict):
    graph = {}
    nodes = list(inputDict.keys())

    # creates dictionary to be copied for each node to map
    # distance to given node
    initInvDict = {node:None for node in nodes}

    for node in nodes:
        graph[node] = initInvDict.copy()

        # deleting the node in its mapping because dist will be zero
        del graph[node][node]
        nodeCoord = inputDict[node]
        for key in graph[node].keys():
            if key != node:
                compareNode = inputDict[key]
                npArray = np.array([nodeCoord, compareNode])
                distance = int(scipy.spatial.distance.pdist(npArray, 'cityblock')[0])
                graph[node][key] = distance
    return graph



def main(argv):
    # Checking all input args are present
    if len(argv) < 1:
        print("You are missing a: Input File, Output File, and Time Limit.")
        print("Please follow form: python tsp-3510.py <input-coordinates.txt>  <output-tour.txt> <time>")
        return
    if len(argv) < 2:
        print("You may be missing a: Output File and Time Limit.")
        print("Please follow form: python tsp-3510.py <input-coordinates.txt>  <output-tour.txt> <time>")
        return
    if len(argv) < 3:
        print("You may be missing a: Time Limit.")
        print("Please follow form: python tsp-3510.py <input-coordinates.txt>  <output-tour.txt> <time>")
        return

    if not argv[2].isdigit():
        print("Time limit must be an integer.")
        print("Please follow form: python tsp-3510.py <input-coordinates.txt>  <output-tour.txt> <time>")
        return

    inputFile = argv[0]
    outputFile = argv[1]
    timeLimit = argv[2]

    start = start = time.time()
    # creates initail mapping from data
    nodeCoordMapping = read_coordinates(inputFile)

    # creates graph
    graph = create_graph(nodeCoordMapping)

    end = time.time()
    print(end - start)

    start = start = time.time()

    nodeCoordMapping = numpy_import(inputFile)

    graph1 = create_graph(nodeCoordMapping)

    end = time.time()
    print(end - start)


    print(graph == graph1)
    # npImport = np.loadtxt(inputFile, delimiter=' ', dtype=float)
    # print(npImport)


if __name__ == "__main__":
   main(sys.argv[1:])
