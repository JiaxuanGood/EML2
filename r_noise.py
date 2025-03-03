from u_base import *
from a_ABC2 import AdaBoostC2
from a_MLDE import MLDE
from a_HEPR import HEPR_noise,mistaking
from c_BOOMER import BOOMER
from c_MLWSE import MLWSE
from c_DECC import DECC

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='data/')
    kb = 20
    misrate = 0.3
    k_this = int(kb*misrate)
    for dataIdx in range(numdata):
        X,Y,Xt,Yt = rd.readData(dataIdx)

        Y2 = mistaking(np.array(Y),misrate,'noise')
        # prediction,traintime,testtime = HEPR_noise(X,Y2,Xt,Yt,k_this)
        # saveResult(datasnames[dataIdx], 'HEPR', evaluate(prediction, Yt), traintime, testtime)

        Y3 = np.array(Y2)/2+0.5

        start_time = time()
        learner = BOOMER()
        learner.train(X,Y3)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'BOOMER', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        start_time = time()
        learner = MLWSE()
        learner.train(X,Y3)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'MLWSE', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        
        start_time = time()
        learner = AdaBoostC2()
        learner.train(X,Y3)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'ABC2', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
