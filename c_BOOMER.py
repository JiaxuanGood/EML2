from mlrl.boosting import Boomer as boom
from u_base import *

class BOOMER():
    def train(self, traX, traY):
        # self.learner = boom(loss='logistic-example-wise',parallel_rule_refinement='false', parallel_statistic_update='false', parallel_prediction='false')
        self.learner = boom(loss='logistic-example-wise')
        # self.learner = boom()
        self.learner.fit(traX, traY)
    def test(self, tesX):
        return self.learner.predict(tesX)

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Yeast","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'BOOMER'

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = BOOMER()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
