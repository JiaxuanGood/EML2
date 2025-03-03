from u_base import *

class AdaboostMH():
    def __init__(self, Q, T, delta=0.01):
        self.Q = Q
        self.T = T
        self.delta = delta
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
        self.rounds = 0
    def induce(self, X, Y):
        Dst_s = np.ones((self.Q,len(X)))   # initial distribution: Uniform distribution
        for round in range(self.T):
            baseLearner = BR()
            baseLearner.train(X, Y, Dst_s)
            alpha,Dst_s = self.hamming(Y, baseLearner.test(X), Dst_s)
            if(alpha==0 or alpha==1):
                break
            self.Alpha_s.append(alpha)
            self.allLearner.append(baseLearner)
        savearray([round+1,alpha], 'log/AdaMH_T')
    def hamming(self, T, prediciton, Dst):
        tmp1 = np.round(prediciton)
        result = np.array(tmp1==T)*1
        error = 1 - np.sum(result*np.transpose(Dst))/(len(T)*self.Q)
        # print(error, sum(result)/len(result))
        # print(np.sum(Dst))
        if(error > 0.5):
            return 1,Dst
        if(error < self.delta):
            return 0,np.ones((self.Q,len(X)))
        alpha = 0.5*np.log((1-error)/error)
        Dst3 = []
        for i in range(self.Q):
            Dst2 = Dst[i]*np.exp(-(result[:,i]-0.5)*2*alpha)
            Dst3.append(self.distribution_adj(Dst2))
        return alpha,Dst3
    def distribution_adj(self, Dst):
        gap = min(Dst)
        if(gap<=0):
            print('dst error!!!')
            Dst = Dst - gap + 0.01
        ssum = sum(Dst)
        Dst = Dst * len(Dst)
        Dst = Dst/ssum
        return Dst
    def test(self, Xt):
        prediction = np.zeros((len(Xt),self.Q))
        for tt in range(len(self.Alpha_s)):
            prediction += self.allLearner[tt].test(Xt) * self.Alpha_s[tt]
        return prediction

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    for dataIdx in range(numdata):
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = AdaboostMH(np.shape(Y)[1], 10)
        learner.induce(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'Adaboost_MH', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
