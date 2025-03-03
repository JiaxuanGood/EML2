from u_base import *
from u_basemulti import *
import math
from sklearn.metrics.pairwise import pairwise_distances,cosine_similarity

def softthres(x,e):
    a=np.maximum(x-e,0)
    b=np.maximum(-1*x-e,0)
    return a-b
def Lasso(X,Y,paras):
    """
    X:the confidence score matrix
    Y:label
    alpha: sparsity parameter
    beta:label correlation parameter
    maxIter: max interation
    """
    alpha,beta,maxIter,miniLossMargin = paras
    XTX=np.dot(np.transpose(X), X)
    XTY=np.dot(np.transpose(X), Y)
    #Initialize the w0,w1
    W_s = np.dot(np.ones(np.shape(XTY)), 0.33)
    W_s_1 = W_s
    # Calculate the similarity distance
    H = pairwise_distances(np.transpose(Y),metric="cosine")

    iter = 1
    oldloss = 0
    # Colculate Lipschitz constant
    Lip = math.sqrt( 2 * math.pow(np.linalg.norm(XTX,ord=2),2) +  math.pow(np.linalg.norm(beta*H ,ord=2), 2) )
    # Initialize b0,b1
    bk=1
    bk_1=1
    # the accelerate proximal gradient
    while iter<=maxIter:
        W_s_k = W_s + np.dot((bk_1 - 1) / bk ,(W_s - W_s_1))
        Gw_s_k = W_s_k - (1 / Lip) * ((np.dot(XTX , W_s_k) - XTY) + beta * np.dot(W_s_k , H))

        bk_1 = bk
        bk = (1 + math.sqrt(4 * math.pow(bk,2) + 1)) / 2
        W_s_1 = W_s
        # soft-thresholding operation
        W_s = softthres(Gw_s_k, alpha / Lip)
        a=np.transpose(np.dot(X,W_s)-Y)
        b=np.dot(X,W_s)-Y

        # Calculate the least squares loss
        predictionLoss=np.trace(np.dot(a,b))
        # Calculate correlation
        correlation=np.trace(np.dot(H,np.dot(np.transpose(W_s),W_s)))
        # Calculate sparsity
        # sparsity=np.sum(np.sum(np.int64(W_s!=0),axis=0),axis=1)
        sparsity=np.sum(np.sum(np.int64(W_s!=0)))
        # Calculate total loss
        totalloss = predictionLoss + beta * correlation + alpha * sparsity

        if math.fabs(oldloss-totalloss) <= miniLossMargin:
            break
        elif totalloss <= 0:
            break
        else:
            oldloss = totalloss
        iter = iter + 1
    return W_s

class MLWSE():
    def __init__(self, paras=[math.pow(10, -4),math.pow(10, -3),200,0.0001]):
        self.paras = paras
        self.bases = []
        self.weights = []
    def train(self, trainX, trainY):
        n_2 = int(len(trainX)*0.5)
        idx = np.arange(len(trainX))
        idx1,idx2 = idx[:n_2],idx[n_2:]
        this_X,this_Y = trainX[idx1],trainY[idx1]
        this_X2,this_Y2 = trainX[idx2],trainY[idx2]
        this_Y = fill1(this_Y)
        this_Y2 = fill1(this_Y2)
        base1,base2,base3 = BR(),CC(),LP()
        base1.train(this_X, this_Y)
        base2.train(this_X, this_Y)
        base3.train(this_X, this_Y)
        self.bases = base1,base2,base3
        new_prediction_BR,new_prediction_CC,new_prediction_LP = self.bases[0].test(this_X2),self.bases[1].test(this_X2),self.bases[2].test(this_X2)
        stacking = np.hstack((new_prediction_BR, new_prediction_CC, new_prediction_LP))  # 3 matrix -> 1 matrix
        self.weights = Lasso(stacking, this_Y2, self.paras)
    def test(self, testX):
        prediction_BR, prediction_CC, prediction_LP = self.bases[0].test(testX),self.bases[1].test(testX),self.bases[2].test(testX)
        new_stacking = np.hstack((prediction_BR, prediction_CC, prediction_LP))
        output = np.dot(new_stacking, self.weights)
        return output

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Image","Yeast","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'MLWSE'

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = MLWSE()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
