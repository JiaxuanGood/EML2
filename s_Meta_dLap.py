import numpy as np
import math
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity
from sklearn.neighbors import NearestNeighbors
from s_utils import normX1

def softthres(x,e):
    a=np.maximum(x-e,0)
    b=np.maximum(-1*x-e,0)
    return a-b

def laplacian(X, k=10):
    findNb = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    findNb.fit(X)
    indices = findNb.kneighbors(X, return_distance=False)
    n = len(X)
    X = normX1(X)
    S = np.zeros((n,n))
    for i in range(n):
        for j in range(n):
            if(j in set(indices[i])):
                S[i,j] = np.exp(-0.5*math.pow(np.linalg.norm((X[i]-X[j])), 2))
                # S[i,j] = 1
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(S[i])
    H = D-S
    return H

def laplacian_Y(Y):
    n = np.shape(Y)[1]
    S = pairwise_distances(np.transpose(Y),metric="cosine")
    D = np.zeros((n,n))
    for i in range(n):
        D[i,i] = np.sum(S[i])
    L = D-S
    return L

#Lasso is implemented using the accelerated proximal gradient 
def meta4(X_org, Xu, Xl,Y,alpha,beta,enta,maxIter,miniLossMargin):
    """
    X:the confidence score matrix
    Y:label
    alpha: sparsity parameter
    beta:label correlation parameter
    enta:initiave w
    maxIter: max interation
    """
    # print(np.shape(Xl),np.shape(Xu))
    X = np.vstack((Xl,Xu))
    XTX=np.dot(np.transpose(Xl), Xl)
    XTY=np.dot(np.transpose(Xl), Y)
    #Initialize the w0,w1
    # W_s = np.dot(np.linalg.inv(XTX + enta * np.eye(n_features)),XTY).astype(np.float)
    W_s = np.dot(np.ones(np.shape(XTY)), 0.33)
    W_s_1 = W_s
    # Calculate the similarity distance
    # H = pairwise_distances(np.transpose(Y),metric="cosine")
    H = laplacian(X_org, k=10)
    H_Y = laplacian_Y(Y)

    iter = 1
    oldloss = 0
    # Colculate Lipschitz constant
    print(np.shape(np.transpose(X)),np.shape(H),np.shape(X))
    Lip = math.sqrt( 2 * math.pow(np.linalg.norm(XTX,ord=2),2) + 
        2 * math.pow(np.linalg.norm(beta*np.dot(np.dot(np.transpose(X), H), X) ,ord=2), 2) + 
        2 * math.pow(np.linalg.norm(H_Y,ord=2), 2))
    # Initialize b0,b1
    bk=1
    bk_1=1
    # the accelerate proximal gradient
    while iter<=maxIter:
        W_s_k = W_s + np.dot((bk_1 - 1) / bk ,(W_s - W_s_1))
        
        Gw_s_k = W_s_k - (1 / Lip) * ((np.dot(XTX , W_s_k) - XTY) + beta * np.dot( np.dot(np.dot(np.transpose(X), H), X), W_s_k) + enta * np.dot(W_s_k, H_Y) ) 
        

        bk_1 = bk
        bk = (1 + math.sqrt(4 * math.pow(bk,2) + 1)) / 2
        W_s_1 = W_s
        # soft-thresholding operation
        W_s = softthres(Gw_s_k, alpha / Lip)

        a=np.transpose(np.dot(Xl,W_s)-Y)
        b=np.dot(Xl,W_s)-Y

        # Calculate the least squares loss
        predictionLoss=np.trace(np.dot(a,b))
        # Calculate correlation
        XW = np.dot(X,W_s)
        correlation=np.trace(np.dot(np.dot(np.transpose(XW), H),XW))
        correlation_Y=np.trace(np.dot(np.dot(W_s, H_Y),np.transpose(W_s)))
        # Calculate sparsity
        sparsity=np.sum(np.sum(np.int64(W_s!=0)))
        # Calculate total loss
        totalloss = predictionLoss + beta * correlation + alpha * sparsity + enta * correlation_Y

        if math.fabs(oldloss-totalloss) <= miniLossMargin:
            break
        elif totalloss <= 0:
            break
        else:
            oldloss = totalloss

        iter = iter + 1

    return W_s
