import numpy as np
from scipy.sparse import csr_matrix
from math import log
from math import inf
from scipy.io import loadmat
import matplotlib.pyplot as plt
import pyro
import pyro.distributions as dist
import torch
import time

def make_grid_graph(h, w):
    nNodes = h * w
    adj = np.zeros((nNodes, nNodes), dtype='int')
    # generate grid graph
    #add down/up edges
    for i in range(nNodes):
        if(i//w==(h-1)):
            continue
        else:
            adj[i, i+w] = 1
            adj[i+w, i] = 1
    #add left/right edge
    for i in range(nNodes):
        if((i+1)%w==0):
            continue
        else:
            adj[i, i+1] = 1
            adj[i+1, i] = 1
    return csr_matrix(adj, dtype='int')

class UGM_makeEdgeStruct:
    def __init__(self, adj, nStates, useMex=True, maxIter='100'):
        self.useMex=useMex
        self.maxIter=int(maxIter)
        self.nNodes = int(adj.shape[0])
        adj_ones = np.argwhere(adj).T
        self.nEdges = int(len(adj_ones[0])/2)
        self.nStates=np.full(shape=self.nNodes, fill_value=int(nStates)).astype('int')

        self.edgeEnds = np.zeros((self.nEdges,2), dtype='int')
        eNum = 0
        for i in range(len(adj_ones[0])):
            a = adj_ones[1,i]
            b = adj_ones[0,i]
            if(a < b):
                self.edgeEnds[eNum,:] = [adj_ones[1, i], adj_ones[0, i]]
                eNum += 1
        self.edgeEnds = self.edgeEnds[self.edgeEnds[:,0].argsort()]

        #generate V, E
        # E[i]: index of edges connected to node i
        # V[i]: the sum of the number of edges connected to node i + 1
        edges = self.edgeEnds.T
        nEdges = edges[0].size
        nNei = np.zeros((self.nNodes, 1), dtype='int')
        nei = np.zeros((self.nNodes, 4), dtype='int')
        for i in range(nEdges):
            n1 = edges[0,i]
            n2 = edges[1,i]
            nNei[n1] = nNei[n1]+1
            nNei[n2] = nNei[n2]+1
            nei[n1, nNei[n1]-1] = i+1
            nei[n2, nNei[n2]-1] = i+1
        edge = 1
        V = np.zeros((self.nNodes+1,1), dtype='int')
        E = []
        for i in range(self.nNodes):
            V[i] = edge
            nodeEdges = np.sort(nei[i][0:int(nNei[i])])
            E=np.concatenate([E, nodeEdges])
            edge = edge + len(nodeEdges)
        V[self.nNodes] = edge
        self.E = np.reshape(E, (2*nEdges,1)).astype('int')
        self.V = V

class UGM_LogConfigurationPotential:
    def __init__(self, y, nodePot, edgePot, edgeEnds):
        nNodes = nodePot.shape[0]
        nEdges = edgeEnds.shape[0]

        logPot = 0
        #Nodes
        for n in range(nNodes):
            logPot += log(nodePot[n,int(y[n])])
        # Edges
        for e in range(nEdges):
            n1 = int(edgeEnds[e,0])
            n2 = int(edgeEnds[e,1])
            logPot += log(edgePot[e][y.item(n1)][y.item(n2)])
        self.logPot = logPot

class UGM_Gibbs_Sample:
    # Gibbs Sampling
    def __init__(self, nodePot, edgePot, edgeStruct, buringIn):
        (nNodes, maxStates) = nodePot.shape
        edgeEnds = edgeStruct.edgeEnds
        V = edgeStruct.V
        E = edgeStruct.E
        nStates = edgeStruct.nStates
        maxIter = edgeStruct.maxIter
        y = np.argmax(nodePot, axis=1)
        samples = []
        start_time = time.time()
        for i in range(buringIn+maxIter):
            if i%100 == 0:
                print("itertaion "+str(i))
                print("--- %s seconds ---" % (time.time() - start_time))
                start_time = time.time()
            for n in range(nNodes):
                pot = nodePot[n, 0:nStates[n]] # Compute Node Potential
                pot = np.reshape(pot, (-1,1))
                edges = E[int(V[n])-1 : int(V[n+1])-1] # Find Neighbors
                flatten_edges = edges.flatten()
                # Multiply Edge Potentials
                for e in flatten_edges:
                    n1 = int(edgeEnds[e-1,0])
                    n2 = int(edgeEnds[e-1,1])
                    if (n == edgeEnds[e-1,0]):
                        ep = edgePot[e-1][0:nStates[n1], y[n2]]
                    else:
                        ep = edgePot[e-1][y[n1], 0:nStates[n2]]
                    ep = np.reshape(ep, (-1,1))
                    pot = pot * ep
                # Samle State
                y[n] = self.sampleDiscrete(pot/np.sum(pot))
            if(i > buringIn):
                samples=np.append(samples, y)
        self.samples = np.reshape(np.array(samples), (-1, nNodes)).astype(dtype='int')
        # get the max potential samples
        nSamples = self.samples.shape[0]
        maxPot = -inf
        for s in range(nSamples):
            logPot = UGM_LogConfigurationPotential(self.samples[s], nodePot, edgePot, edgeEnds)
            if logPot.logPot > maxPot:
                maxPot = logPot.logPot
                self.pred = self.samples[s]


    def sampleDiscrete(self, p):
        test = torch.tensor(p).T
        out=torch.multinomial(test[0],1,replacement=True)
        return out.item()
        #U = np.random.rand()
        #u = 0
        #for i in range(len(p)):
        #    u = u + p[i]
        #    if u > U:
        #        return i
        #return len(p)-1
class MRF:
    '''
    input: prob_map: [row X col X classes]
    output: images [row X col]
    '''
    def __init__(self, prob_map, Xv, Yv):
        #self.params = [0.01, 0.1, 1, 10]
        self.params = [10]
        (self.nRows, self.nCols, self.nClasses) = prob_map.shape
        self.nNodes = self.nRows*self.nCols
        #generate grid
        print("make_grid_graph")
        adj = make_grid_graph(self.nRows, self.nCols)
        maxPerf = 0
        for param in self.params:
            print("params: "+str(param))
            out_map = self.MRFpred(prob_map, adj, param)
            out_map = out_map.flatten()
            print("debug0: out_map 154")
            print(out_map.shape)
            print(Xv.shape)
            print(Yv.shape)
            print(out_map)
            Yout = np.take(out_map, Xv, axis=0)
            perf = np.sum(Yout == Yv)

            print(Yout)
            print(perf)

            if perf > maxPerf:
                maxPerf = perf
                self.selParams = param
                self.outMap = out_map
        print(maxPerf/len(Yout))

    def MRFpred(self, prob_map, adj, beta):
        # nodePot is the data cost, now it is a constant value
        # TODO modified to parallel compute tiles
        nodePot=prob_map.reshape((self.nNodes, self.nClasses))
        edgeStruct=UGM_makeEdgeStruct(adj, self.nClasses, 0)
        #initial edge value
        edgePot=np.zeros((edgeStruct.nEdges, self.nClasses, self.nClasses))
        for i in range(edgeStruct.nEdges):
            n1 = edgeStruct.edgeEnds[i,0]
            n2 = edgeStruct.edgeEnds[i,1]
            diffPot = beta
            edgePot[i,:,:]= np.exp(-diffPot*(1-np.eye(self.nClasses)))
        gibbs=UGM_Gibbs_Sample(nodePot,edgePot, edgeStruct, 1000)
        # Compute Max Potential for the Samples
        pred=gibbs.pred.reshape((self.nRows, self.nCols))
        return pred


    def display(self):
        print(self.nRows)


prob_map = loadmat('./data/prob_map.mat')
Xv = loadmat('./data/MRF800_XvalidC_numTrainEgL_1_numOfTrials_8.mat')
Yv = loadmat('./data/MRF800_Yvalid_numTrainEgL_1_numOfTrials_8.mat')
Xt = loadmat('./data/MRF800_XtestC_numTrainEgL_1_numOfTrials_8.mat')
Yt = loadmat('./data/MRF800_Ytest_numTrainEgL_1_numOfTrials_8.mat')
Xv = Xv['XvalidC'].astype('int')
Yv = Yv['Yvalid'].astype('int')
Xt = Xt['XtestC'].astype('int')
Yt = Yt['Ytest'].astype('int')
Xv0 = np.sum(Xv, axis=1)-1
Yv0 = Yv.flatten()-1
Xt0 = np.sum(Xt, axis=1)-1
Yt0 = Yt.flatten()-1

#Pred = np.zeros((3,3,2))
#Pred[:,:,0] = [[0.1, 0.2, 0.3],
#               [0.7, 0.6, 0.5],
#               [0.6, 0.4, 0.2]]
#Pred[:,:,1] = [[0.4, 0.4, 0.5],
#               [0.6, 0.6, 0.7],
#               [0.3, 0.8, 0.3]]
#Xv1 = np.array([0, 3])
#Yv1 = np.array([1, 0])

mrf=MRF(prob_map['prob_map'], Xv0, Yv0)
outMap = mrf.outMap
Ypred = np.take(outMap.flatten(), Xt.sum(axis=1), axis=0)
OA = np.count_nonzero(Ypred == Yt)/Ypred.shape[0]
print(Ypred.shape)
print(Yt.shape)
print(OA)

#mrf = MRF(pred, Xv1, Yv1)
#print(mrf.outMap)
#print(mrf.selParams)
