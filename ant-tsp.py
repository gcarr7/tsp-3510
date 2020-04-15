import sys
from pprint import pprint
import scipy.spatial.distance
import numpy as np
import time
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
            currentAnt.total_cost += self.graph.matrix[currentAnt.visitedNodes[-1]][currentAnt.visitedNodes[0]]

            self.antList.append(currentAnt)


    def opt_path_finder(self):
        path = list()
        cost = float("inf")
        for num in range(self.generations):
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
        self.visitedNodes = [self.start]
        self.unvisitedNodes = [index for index in range(self.graph.size)]
        self.unvisitedNodes.remove(self.start)
        self.current = self.start

    def create_eta(self):
        ones = np.ones((self.graph.size, self.graph.size), dtype=int)
        divided = np.divide(ones, self.graph.matrix)
        np.fill_diagonal(divided, 0)
        self.eta = divided


    def get_next(self):
        den = 0
        for node in self.unvisitedNodes:
            den += self.graph.pheromone_matrix[self.current][node] ** self.colony.alpha * self.eta[self.current][node] ** self.colony.beta
        probabilities = np.zeros(self.graph.size)

        for index in range(self.graph.size):
            if index in self.unvisitedNodes:
                probabilities[index] = self.graph.pheromone_matrix[self.current][index] ** self.colony.alpha * \
                                    self.eta[self.current][index] ** self.colony.beta / den
        selected = 0
        rand = random.random()

        for index in range(len(probabilities)):
            rand -= probabilities[index]
            if rand <=0:
                selected = index
                break

        self.unvisitedNodes.remove(selected)
        self.visitedNodes.append(selected)
        self.total_cost += self.graph.matrix[self.current][selected]
        self.current = selected

    def delta_update(self):
        for index in range(1, len(self.visitedNodes)):
            row = self.visitedNodes[index - 1]
            col = self.visitedNodes[index]
            if self.colony.strategy == 1:
                self.pheromone_delta[row][col] = self.colony.q
            elif self.colony.strategy == 2:
                self.pheromone_delta[row][col] = self.colony.q / self.graph.matrix[i][j]
            else:
                self.pheromone_delta[row][col] = self.colony.q / self.total_cost



def main(argv):
    warnings.filterwarnings("ignore")

    inputFile = argv[0]
    # outputFile = argv[1]
    # timeLimit = argv[2]


    # costList = list()
    # for num in range(10):
    #     print("Run: " + str(num))
    #     graph = Graph(inputFile)

    #     graph.read_coordinates()
    #     graph.create_cost_matrix()
    #     graph.create_pheromone_matrix()

    #     colony = Colony(graph, 40, 400, .5, 10.0, .3, 10, 1)
    #     path, cost = colony.opt_path_finder()
    #     costList.append(cost)

    # print(costList)
    # print(mean(costList))

    graph = Graph(inputFile)

    graph.read_coordinates()
    graph.create_cost_matrix()
    graph.create_pheromone_matrix()

    colony = Colony(graph, 2, 2, .5, 10.0, .3, 10, 1)
    path, cost = colony.opt_path_finder()


    # path, cost = colony.opt_path_finder()
    print(path)
    print(cost)


if __name__ == "__main__":
   main(sys.argv[1:])