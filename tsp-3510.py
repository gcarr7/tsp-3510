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

def getSets(size, nodes, set_mem):
    current_sets = []
    if size==2 or set_mem==[]:
        set_mem = [{1.0,x} for x in nodes.keys()]
        set_mem.remove({1.0})
    else:
        for j in current_sets:
          print(j)
          for i in range(2, len(nodes.keys())):
               set_mem.append(j.add(i))
               current_sets.append(j.add(i))
    return [current_sets, set_mem]


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

    start = time.time()
    # creates initail mapping from data
    nodeCoordMapping = read_coordinates(inputFile)

    # creates graph
    graph = create_graph(nodeCoordMapping)

    end = time.time()
    print(end - start)

    #Dynamic Programing.

    s = [1]      #add to this list as we add in nodes (always start at 1)
    cost = {}  # matrix to track min const using dynamic proraming
    all_nodes = graph.keys()

    set_mem = []
    for k in range(2, len(all_nodes)+1):
        sets = getSets(k, graph, set_mem)
        set_mem = sets[1]
        for T in sets[0]:
            for i in (all_nodes - T):
               if (len(T) == 2):
                    cost[tuple(T),i] = graph[1][i]
               if (len(T) > 2):
                    cost[T,i] = float("inf")
                    #min_tracker = 0
                    for j in (T-[1, i]) :
                         if ((cost[T-{i},j] + graph[j][i]) < cost[T,i]):
                              cost[T,i] = cost[T-{i},j] + graph[j][i]
                              min_tracker = j
                    #s += [min_tracker]
            #s.remove(i)

    print(cost)
    print(s)



if __name__ == "__main__":
   main(sys.argv[1:])
