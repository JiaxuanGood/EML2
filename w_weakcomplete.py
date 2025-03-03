from sklearn.neighbors import NearestNeighbors
import numpy as np

'''semi-completion: simply sum(neighbor)/k'''
def completeLabel2(X, Y, k=10, complete_which=0):
    findNb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(X, return_distance=False)
    label = np.array(Y).astype(np.float16)
    for i in range(len(Y)):
        indx = indices[i]
        for j in range(len(Y[0])):
            if(Y[i][j]==complete_which):
                label[i][j]=np.sum(Y[indx,j])/k
    return label
'''semi-completion: sum(neighbor)/[k-num(mis)]'''
def completeLabel(X, Y, k=10, complete_which=0, self_weight=0):
    findNb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(X, return_distance=False)
    label = np.array(Y).astype(np.float16)
    for i in range(len(Y)):
        indx = indices[i]
        for j in range(len(Y[0])):
            if(complete_which==2):
                label[i][j]=np.sum(Y[indx,j]+label[i][j]*self_weight)/(k+self_weight)
            if(Y[i][j]==complete_which):
                k_this = np.sum(np.abs(Y[indx,j]))
                if(k_this==0):
                    label[i][j]=complete_which
                else:
                    label[i][j]=np.sum(Y[indx,j]+complete_which*self_weight)/(k_this+self_weight)
    return label

'''convert numeric label to int label: 1 for positive, 0 for negative'''
def intlabel(labelorg, threshold=0.5):
    N,Q = np.shape(labelorg)
    label = np.zeros((N,Q))
    for i in range(N):
        for j in range(Q):
            if(labelorg[i][j]>=threshold):
                label[i][j]=1
            else:
                label[i][j]=0
    return label