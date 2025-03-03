from sklearn.metrics.pairwise import pairwise_distances
import numpy as np
from w_weaksample import *

# def getSimMat(label):
#     label = np.transpose(label)
#     cosmat = pairwise_distances(label, metric='cosine') # smaller, better
#     return cosmat

def entropy(lst):   # bigger, better
    N = len(lst)
    ent = 0
    for i in range(4):
        freq = np.sum(lst==i) / N
        ent += freq*freq
    return 1-ent

def getEntMat(label):
    label = np.transpose(intlabel(label))
    Q = len(label)
    entmat = np.zeros((Q,Q))
    for i in range(Q):
        for j in range(i+1,Q):
            tmp = label[i]+2*label[j]
            entmat[i][j] = -entropy(tmp)
            entmat[j][i] = entmat[i][j]
        entmat[i][i] = 1
    return entmat

'''similarity'''
def ray(data, label1,label2):   # bigger, better
    thislabel = label1 - label2
    idx_pos = np.argwhere(thislabel<0).flatten()
    idx_neg = np.argwhere(thislabel>0).flatten()
    data_pos = data[idx_pos]
    data_neg = data[idx_neg]
    cent_pos = np.sum(data_pos, 0)/len(data_pos)
    cent_neg = np.sum(data_neg, 0)/len(data_neg)
    sb = np.sum((cent_pos-cent_neg) * (cent_pos-cent_neg))
    sw = 0
    for a in data_pos:
        sw += np.sum((a-cent_pos)*(a-cent_pos))
    for b in data_neg:
        sw += np.sum((b-cent_neg)*(b-cent_neg))
    return sb/sw

def getRayMat(data, label):
    label = np.transpose(label)
    Q = len(label)
    raymat = np.zeros((Q,Q))
    for i in range(Q):
        for j in range(i+1,Q):
            raymat[i][j] = -ray(data, label[i], label[j])
            raymat[j][i] = raymat[i][j]
        raymat[i][i] = 1
    return raymat

def prime(G, begin=0):
    N = len(G)
    INF = 9999999
    selected_node = np.zeros(N)
    no_edge = 0
    selected_node[begin] = True
    edgs = []
    while (no_edge < N - 1):
        minimum = INF
        a = begin
        b = begin
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        # not in selected and there is an edge
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n
        edgs.append([a,b])
        selected_node[b] = True
        no_edge += 1
    return edgs

def prime_ring(G_org, begin=0):
    G = np.array(G_org)
    N = len(G)
    INF = 9999999
    selected_node = np.zeros(N)
    no_edge = 0
    selected_node[begin] = True
    edgs = []
    while (no_edge < N - 1):
        minimum = INF
        a = begin
        b = begin
        for m in range(N):
            if selected_node[m]:
                for n in range(N):
                    if ((not selected_node[n]) and G[m][n]):  
                        if minimum > G[m][n]:
                            minimum = G[m][n]
                            a = m
                            b = n
        edgs.append([a,b])
        G[a,:]=1
        G[:,a]=1
        selected_node[b] = True
        no_edge += 1
    return edgs

def primes(G, begins, mode=0):
    num = len(begins)
    edgeLists = []
    for i in range(num):
        if(mode==0):
            this_list = prime(G,begins[i])
        else:
            this_list = prime_ring(G,begins[i])
        edgeLists.append(this_list)
        for j in range(len(this_list)):
            G[this_list[j][0]][this_list[j][1]] = 1
    return edgeLists
    
if __name__=='__main__':
    # G = [[0, 19, 5, 0, 0],
    #     [19, 0, 5, 9, 2],
    #     [5, 5, 0, 1, 6],
    #     [0, 9, 1, 0, 1],
    #     [0, 2, 6, 1, 0]]
    # edgs = prime(G)
    # print(edgs)
    datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","Genbase","GnegativeGO","GpositiveGO","Image",
        "Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess","Coffee","VirusGO","Yeast","Yelp"]
    rd = ReadData(datas=datasnames)
    for dataIdx in range(1):
        X,Y,Xt,Yt = rd.readData(dataIdx)
        N,Q = np.shape(Y)
        M = np.shape(Yt)[0]
        print(N,Q,M)
        Y = mistaking(Y,0.3)
        Y_new = completeLabel(X,Y)
        # simMat = getRayMat(Y_new)
        # print(simMat)
        raymat = getRayMat(X,Y_new)
        print(raymat)
        # print(np.sum(simMat,0))
        print(prime_ring(raymat,0))
        edgeLists = primes(raymat, [0,2], 1)
        print(edgeLists[0])
        print(edgeLists[1])
