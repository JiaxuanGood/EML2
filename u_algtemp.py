from u_mReadData import ReadData
from time import time
from u_evaluation import evaluate
from u_savedata import saveResult
import numpy as np

class alg():
    def train(self, traX, traY):
        pass
    def test(self, tesX):
        pass

numdata = 1
datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
# rd = ReadData(datas=datasnames,genpath='data/')
algname = 'ALG'

'''k-fold'''
for dataIdx in range(numdata):
    print(dataIdx)
    k_fold,X_all,Y_all = rd.readData_CV(dataIdx)
    for train, test in k_fold.split(X_all, Y_all):
        X = X_all[train]
        Y = Y_all[train]
        Xt = X_all[test]
        Yt = Y_all[test]
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = alg()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

'''k-fold with 1 result'''
n_fold = 10
for dataIdx in range(numdata):
    print(dataIdx)
    tmp_rst = np.zeros(13)
    k_fold,X_all,Y_all = rd.readData_CV(dataIdx,n_fold)
    for train, test in k_fold.split(X_all, Y_all):
        X = X_all[train]
        Y = Y_all[train]
        Xt = X_all[test]
        Yt = Y_all[test]
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = alg()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        tmp_rst += np.array(np.append(evaluate(prediction, Yt),[mid_time-start_time, time()-mid_time]))
    saveResult(datasnames[dataIdx], algname, tmp_rst/n_fold)

'''train-test'''
for dataIdx in range(numdata):
    print(dataIdx)
    X,Y,Xt,Yt = rd.readData(dataIdx)
    print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
    # start_time = time()
    # learner = alg()
    # learner.train(X,Y)
    # mid_time = time()
    # prediction = learner.test(Xt)
    # saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
