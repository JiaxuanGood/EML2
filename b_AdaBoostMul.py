from u_base import *

class adaboost():
    def __init__(self, T=10):
        self.T = T
    def induce(self, X, Y_a, Xt):
        Dst = np.ones(len(X))
        Alphas = []
        Learners = []
        for t in range(self.T):
            learner = self.train(X, Y_a, Dst)
            Learners.append(learner)
            alpha,Dst = self.boosting(X, Y_a, Dst, learner)
            if(alpha==0):
                break
            Alphas.append(alpha)
        if(len(Alphas)==0):
            Alphas.append(1)
        Alphas = np.array(Alphas)/len(Alphas)
        print(Alphas)
        prediction = np.zeros(len(Xt))
        thisT = len(Alphas)
        for t in range(thisT):
            prediction += np.array(Learners[t].predict_proba(Xt))[:,1] * Alphas[t]
        return np.array(prediction/thisT)
    def train(self, X, Y_a, Dst):
        singleLearner = base_cls()
        gap = min(Dst)
        if(gap<=0):
            Dst = Dst - gap + 0.01
        singleLearner.fit(X, Y_a, sample_weight=Dst)
        return singleLearner
    def boosting(self, X, Y_a, Dst, learner):
        result = np.array(learner.predict(X)!=Y_a)
        error = sum(result*Dst)/len(X)
        print(error, end='\t')
        alpha = 0.5*np.log((1-error)/error)
        if(error<0.0001):
            return 0,np.zeros(len(X))
        Dst2 = Dst*np.exp((result-0.5)*2*alpha)
        Dst2 = Dst2/sum(Dst2)
        return alpha,Dst2
    
if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        
        start_time = time()
        prediction = []
        for q in range(np.shape(Y)[1]):
            boostLearner = adaboost()
            prd = boostLearner.induce(X, Y[:,q], Xt)
            prediction.append(prd)
        prediction = np.transpose(prediction)
        saveResult(datasnames[dataIdx], 'AdaBoost.BR.Mul', evaluate(prediction, Yt), (time()-start_time))
