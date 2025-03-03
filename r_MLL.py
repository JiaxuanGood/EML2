from u_base import *
from a_ABC2 import AdaBoostC2
from a_MLDE import MLDE
from c_BOOMER import BOOMER
from c_MLWSE import MLWSE
from c_DECC import DECC

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='data/')
    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        
        start_time = time()
        learner = AdaBoostC2()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'ABC2', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        weights = learner.get_alphaweights()
        numRounds = np.sum(weights!=0,0)
        savearray(numRounds,'log/ABC2_T')

        start_time = time()
        learner = MLDE()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'MLDE', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        start_time = time()
        learner = BOOMER()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'BOOMER', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        start_time = time()
        learner = MLWSE()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'MLWSE', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

        start_time = time()
        learner = DECC()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'DECC', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
