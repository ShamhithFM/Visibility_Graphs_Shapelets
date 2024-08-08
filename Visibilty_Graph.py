import numpy as np
from ts2vg import NaturalVG
from ts2vg import HorizontalVG
from Params import Params
import pandas as pd

def doNVG(timeSeries):
    nvg = NaturalVG()
    network = nvg.build(timeSeries).as_networkx()
    adjacencyMatrix = getAdjacencyMatrix(network)
    params = Params(network, normalizedGraphEntropy(network), adjacencyMatrix)

    newCsvRow = str(params)
    return newCsvRow


def getAdjacencyMatrix(network):
    numberOfNodes = len(network.nodes)
    adjacencyMatrix = np.zeros((numberOfNodes, numberOfNodes))
    for edge in network.edges:
        i, j = edge
        adjacencyMatrix[i][j] = 1
        adjacencyMatrix[j][i] = 1
    return adjacencyMatrix


def normalizedGraphEntropy(graph):
    # Get the degree distribution of the graph
    degreeSequence = [d for n, d in graph.degree()]

    # Compute the probability distribution of degrees
    degreeCounts = np.bincount(degreeSequence)
    degreeCounts = [i for i in degreeCounts if i != 0]
    degreeProbs = [degreeCount/ len(degreeSequence) for degreeCount in degreeCounts]

    # Compute entropy
    entropy = -np.sum(degreeProbs * np.log2(degreeProbs))

    # Normalize entropy
    maxEntropy = np.log2(len(degreeProbs))
    normalizedEntropy = entropy / maxEntropy

    return normalizedEntropy

def sliding_window(data, window_size):  
    shape = (data.shape[0] - window_size + 1, window_size)  
    strides = (data.strides[0], data.strides[0])  
    return np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)  
  
df = pd.read_excel(r'C:\Users\kamasanis\OneDrive - FM Global\Train.xlsx')  
row1 = df['Gearboxload'].values  
row2 = df['Gearbox+Load on plate'].values  
row3 = df['Faulted rotor bearing'].values  
X = np.row_stack((row1, row2, row3))  
y = np.array([1, 2, 3])  
window_size = 512  
X1 = []  
  
for i in range(X.shape[0]):  
    windows = sliding_window(X[i], window_size)  
    X1.append([doNVG(w) for w in windows])  
  
print(X1)