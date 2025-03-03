from u_base import *
from s_utils import *
from a_SCTM import SCTM
from c_MLWSE import MLWSE

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='data/')

    '''k-fold with 1 result'''
    r_unlabeled = [1,4]
    for dataIdx in range(14):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for z in r_unlabeled:
            Xl,Yl,Xu,Yu = datasplit(X,Y, 1/(1+z))

            start_time = time()
            learner = SCTM(tau=[0.7,0.7])
            learner.train(Xl,Yl,Xu)
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            start_time = time()
            learner = MLWSE()
            learner.train(Xl,Yl)
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'MLWSE_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
