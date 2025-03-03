from u_base import *
from s_feature import *
from s_utils import *
from b_SCTMMetas import meta0

numdata = 14
datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]

rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
# rd = ReadData(datas=datasnames,genpath='data/')
algname = 'SCTM_'

'''k-fold with 1 result'''
r_unlabeled = [1,4]
for dataIdx in range(10,numdata):
    print(dataIdx)
    X,Y,Xt,Yt = rd.readData(dataIdx)
    for z in r_unlabeled:
        Xl,Yl,Xu,Yu = datasplit(X,Y, 1/(1+z))
        for mode in ('lower','upper'):
            start_time = time()
            numbase = 3
            numlabel = np.shape(Y)[1]
            if(mode=='lower'):
                idxset = getidxset_dask(Xl,Yl,numbase)
                Xls = getsubX(Xl, idxset)
            elif(mode=='upper'):
                idxset = getidxset_dask(X,Y,numbase)
                Xls = getsubX(X, idxset)
                Yl = Y
            else:
                pass
            print(np.shape(Xl),np.shape(Xt),np.shape(Yl),np.shape(Yt))
            Xts = getsubX(Xt, idxset)
            mLearners = []
            for i in range(numbase):
                thislearner = BR()
                thislearner.train(Xls[i],Yl)
                mLearners.append(thislearner)
            new_Xl = []
            for i in range(numbase):
                new_Xl.append(mLearners[i].test(Xls[i]))
            new_Xl = np.hstack(tuple(new_Xl))
            '''
            alpha: L1
            beta: manifold
            enta: label correlation
            '''
            W = meta0(new_Xl, Yl, 0.1, 0.01, 200, 0.0001) # stacking only

            mid_time = time()
            new_Xt = []
            for i in range(numbase):
                new_Xt.append(mLearners[i].test(Xts[i]))
            new_Xt = np.hstack(tuple(new_Xt))
            prediction = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'SCTM_'+str(z)+mode, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
