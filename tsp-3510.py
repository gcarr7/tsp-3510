import sys
from pprint import pprint
import scipy.spatial.distance
import numpy as np
import time as tm
import random
import operator
import matplotlib.pyplot as plt
import warnings
from statistics import mean

class Graph():

    def __init__(self, inputFile):
        self.inputFile = inputFile
        self.nodes = None
        self.nodeMapping = None
        self.matrix = None
        self.pheromone_matrix = None
        self.size = None

    def read_coordinates(self):
        nodeCoordMapping = {}
        rawText = open(self.inputFile, 'r').readlines()
        for line in rawText:
            rowList = line.split(' ')
            rowList[2] = rowList[2].replace('\n', '')
            rowList = [round(float(item)) for item in rowList]
            npArray = np.array([rowList[1], rowList[2]])
            nodeCoordMapping[rowList[0]] = npArray
        self.nodeMapping = nodeCoordMapping
        self.size = len(self.nodeMapping)
        self.nodes = list(self.nodeMapping.keys())

    def create_cost_matrix(self):
        tempMatrix = np.zeros((self.size, self.size), dtype=int)

        rowIndex = 0
        for node in self.nodes:
            colIndex = 0
            for otherNode in self.nodes:
                if node != otherNode:
                    distance = int(scipy.spatial.distance.euclidean(self.nodeMapping[node],self.nodeMapping[otherNode]))
                    tempMatrix[rowIndex][colIndex] = distance
                colIndex += 1
            rowIndex += 1
        self.matrix = tempMatrix

    def create_pheromone_matrix(self):
        tempMatrix = np.ones((self.size, self.size))
        intialValue = 1 / (self.size ** 2)
        tempMatrix = tempMatrix * intialValue
        self.pheromone_matrix = tempMatrix

class Colony():

    def __init__(self, graph, ants, generations, alpha, beta, rho, q, strategy):
        self.graph = graph
        self.ants = ants
        self.generations = generations
        self.alpha = alpha
        self.beta = beta
        self.rho = rho
        self.q = q
        self.strategy = strategy
        self.antList = None

    def pheromone_update(self):
        self.graph.pheromone_matrix *= self.rho

        for item in self.antList:
            self.graph.pheromone_matrix += item.pheromone_delta

    def travel(self):
        self.antList = []
        for num in range(self.ants):
            currentAnt = Ant(self, self.graph)
            for num in range(self.graph.size - 1):
                currentAnt.create_eta()
                currentAnt.get_next()
            currentAnt.total_cost += self.graph.matrix[currentAnt.visitedIndices[-1]][currentAnt.visitedIndices[0]]
            currentAnt.visitedNodes.append(currentAnt.visitedNodes[0])
            self.antList.append(currentAnt)


    def opt_path_finder(self, time):
        path = list()
        cost = float("inf")
        start = tm.time()
        for num in range(self.generations):
            if tm.time() - start >= time:
                return path, cost
            self.travel()
            for ant in self.antList:
                if ant.total_cost < cost:
                    cost = ant.total_cost
                    path = ant.visitedNodes
                ant.delta_update()
            self.pheromone_update()
        return path, cost



class Ant():

    def __init__(self, colony, graph):
        self.colony = colony
        self.graph = graph
        self.pheromone_delta = np.zeros((self.graph.size, self.graph.size))
        self.eta = None
        self.total_cost = 0.0
        self.start = random.randint(0, graph.size - 1)
        self.visitedIndices = [self.start]
        # add one to get actual node id
        self.visitedNodes = [self.start + 1]
        self.unvisitedIndices = [index for index in range(self.graph.size)]
        self.unvisitedIndices.remove(self.start)
        self.current = self.start

    def create_eta(self):
        ones = np.ones((self.graph.size, self.graph.size), dtype=int)
        divided = np.divide(ones, self.graph.matrix)
        np.fill_diagonal(divided, 0)
        self.eta = divided


    def get_next(self):
        den = 0
        for node in self.unvisitedIndices:
            den += self.graph.pheromone_matrix[self.current][node] ** self.colony.alpha * self.eta[self.current][node] ** self.colony.beta
        probabilities = np.zeros(self.graph.size)

        for index in range(self.graph.size):
            if index in self.unvisitedIndices:
                probabilities[index] = self.graph.pheromone_matrix[self.current][index] ** self.colony.alpha * \
                                    self.eta[self.current][index] ** self.colony.beta / den
        selected = 0
        rand = random.random()

        for index in range(len(probabilities)):
            rand -= probabilities[index]
            if rand <=0:
                selected = index
                break

        self.unvisitedIndices.remove(selected)
        self.visitedIndices.append(selected)

        # add one to get actual node id
        self.visitedNodes.append(selected + 1)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def delta_update(self):
        for index in range(1, len(self.visitedIndices)):
            row = self.visitedIndices[index - 1]
            col = self.visitedIndices[index]
            if self.colony.strategy == 1:
                self.pheromone_delta[row][col] = self.colony.q
            elif self.colony.strategy == 2:
                self.pheromone_delta[row][col] = self.colony.q / self.graph.matrix[i][j]
            else:
                self.pheromone_delta[row][col] = self.colony.q / self.total_cost

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
                distance = scipy.spatial.distance.euclidean(nodeCoord, compareNode)
                graph[node][key] = distance
    return graph



def main(argv):
    warnings.filterwarnings("ignore")

    inputFile = argv[0]
    outputFile = argv[1]
    timeLimit = argv[2]

    startTime = tm.time()

    #start timer

    graph = Graph(inputFile)
    graph.read_coordinates()
    graph.create_cost_matrix()
    graph.create_pheromone_matrix()

    #costList = list()
    #for num in range(10):
    #    print("Run: " + str(num))

    graphTime = tm.time() - startTime
    if graphTime >= int(timeLimit):
        print("Time out Error: During Graph Development.")
        return
    colony = Colony(graph, 20, 100, .5, 10.0, .3, 10, 1)
    colonyTime = tm.time() - (graphTime + startTime)
    totalTime = colonyTime + graphTime
    if totalTime >= int(timeLimit):
        print("Time out Error: During Colonly Development.")
        return
    path, cost = colony.opt_path_finder(int(timeLimit) - totalTime)

    if len(path) < len(graph.nodes):
        print("Time out Error: Try adjusting colony parameters.")
    #costList.append(cost)
    algoTime = tm.time() - (totalTime + startTime)
    totalTime += algoTime

    #print(costList)
    print(path)
    print(cost)

    startAtOne = []
    curr = path.index(1)

    for node in path:
        startAtOne.append(path[curr])
        if curr == len(path)-1:
            curr = 1
        else:
            curr += 1

    time = tm.time() - startTime

    file = open(outputFile, "a")
    file.write("\n\nRaw Path: {}".format(path))
    file.write("\nPath from 1: {}".format(startAtOne))
    file.write("\nCost: {}".format(cost))
    file.write("\nTotal Time: {} seconds".format(time))
    # can print more time info like graphtime, 
    # algotime and colonytime as needed


    # graph = Graph(inputFile)

    # graph.read_coordinates()
    # graph.create_cost_matrix()
    # graph.create_pheromone_matrix()

    # colony = Colony(graph, 2, 2, .5, 10.0, .3, 10, 1)
    # path, cost = colony.opt_path_finder()

    # dictmap = create_graph(graph.nodeMapping)

    # # path, cost = colony.opt_path_finder()
    # print(path)

    # print(len(path))
    # print(len(set(path)))

    # print(cost)


if __name__ == "__main__":
   main(sys.argv[1:])