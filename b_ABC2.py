from u_base import *

class AdaBoostC2():
    def __init__(self, T=10, delta=0.01, base=''):
        self.T = T
        self.delta = delta
        self.allLearner = []    #(T,Q)
        self.Alpha_s = []   #(T,Q)
        self.Order_s = []   #(T,Q)
        self.base = base
    def train(self, X, Y):
        self.Q = np.shape(Y)[1]
        Dst_s = np.ones((self.Q, len(X)))   # initial distribution: Uniform distribution
        order = randorder(self.Q)   # initial order of classifiers chain: random
        ok = [] # the indexes of exactly classificated labels
        for t in range(self.T):
            Dst_s,error_s = self.trainCC(X, Y, Dst_s, order, ok)
            ok = np.argwhere(np.array(error_s)<self.delta).flatten()
            order = np.argsort(error_s)
            order2 = randorder2(len(order),ok)
            order[len(ok):] = order2
    def trainCC(self, X, Y, Dst_s, order, ok):
        self.Order_s.append(order)
        order = order[len(ok):]
        X_train = np.array(X)
        if(len(ok)>0):
            for q in ok:
                X_train = np.hstack((X_train, Y[:,[q]]))
        Alpha = ['']*self.Q
        baseLearner = ['']*self.Q
        Dst_s2 = ['']*self.Q
        error_s = np.zeros(self.Q)
        for qq in order:
            singleLearner = Baser(self.base)
            singleLearner.fit(X_train, Y[:,qq], Dst_s[qq])
            baseLearner[qq] = singleLearner
            alpha,Dst2,error = self.boosting_a(X_train, Y[:,qq], Dst_s[qq], singleLearner)
            Alpha[qq] = alpha
            Dst_s2[qq] = Dst2
            error_s[qq] = error
            X_train = np.hstack((X_train, Y[:,[qq]]))
        self.allLearner.append(baseLearner)
        self.Alpha_s.append(Alpha)
        return Dst_s2, error_s
    def boosting_a(self, X, Y_a, Dst, learner):
        tmp1 = np.int32(np.round(learner.predict_proba(X)[:,1]))
        result = np.array(tmp1!=Y_a)
        error = sum(result*Dst)/len(X)
        if(error>0.5):
            return 0,Dst,error
        if(error < self.delta):
            return 0,np.ones(len(X)),error #np.ones(len(X))
        alpha = 0.5*np.log((1-error)/error)
        Dst2 = Dst*np.exp(-(result-0.5)*2*alpha)
        Dst3 = self.distribution_adj(Dst2)
        return alpha,Dst3,error
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
        Alpha_weights = self.get_alphaweights()
        prediction = np.zeros((self.Q,len(Xt)))
        prediction_aLabel = ['']*self.Q
        for tt in range(self.T):
            Xt_train = np.array(Xt)
            prediction_t = np.zeros((self.Q,len(Xt)))
            for qq in self.Order_s[tt]:
                if(Alpha_weights[tt][qq]==0):
                    Xt_train = np.hstack((Xt_train, np.reshape(prediction_aLabel[qq], (-1, 1))))
                    continue
                prediction_a = self.allLearner[tt][qq].predict_proba(Xt_train)[:,1]
                prediction_aLabel[qq] = prediction_a
                prediction_t[qq] = np.array(prediction_a) * Alpha_weights[tt][qq]
                if(Alpha_weights[tt][qq]<0):
                    prediction_t[qq] = -prediction_t[qq]
                Xt_train = np.hstack((Xt_train, np.reshape(prediction_a, (-1, 1))))
            prediction = prediction + np.array(prediction_t)
        return np.transpose(prediction)
    def get_alphaweights(self): #adjust weight
        Alpha_weights = np.zeros((self.T, self.Q))
        for i in range(self.T):
            for j in range(self.Q):
                if(self.Alpha_s[i][j]!=''):
                    Alpha_weights[i,j] = self.Alpha_s[i][j] # equal weight: Alpha_weights[i,j] = 1
        for j in range(self.Q):
            if(Alpha_weights[0,j] == 0):
                Alpha_weights[0,j] = 1
        for j in range(self.Q):
            Alpha_weights[:,j] = Alpha_weights[:,j]/sum(np.abs(Alpha_weights[:,j])) # adjusting
        return Alpha_weights

def randorder2(Q, ok):
    lst = randorder(Q)
    for i in range(len(ok)):
        lst[np.argwhere(lst==ok[i]).flatten()] = -1
    lst2 = []
    for i in range(len(lst)):
        if(lst[i]!=-1):
            lst2.append(lst[i])
    return lst2

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
        learner = AdaBoostC2(base='bayes')
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'ABC2_NB', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        weights = learner.get_alphaweights()
        numRounds = np.sum(weights!=0,0)
        savearray(numRounds,'log/ABC2_T')

        start_time = time()
        learner = AdaBoostC2(base='dt')
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'ABC2_DT', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        weights = learner.get_alphaweights()
        numRounds = np.sum(weights!=0,0)
        savearray(numRounds,'log/ABC2_T')

        start_time = time()
        learner = AdaBoostC2(base='lr')
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'ABC2_LR', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        weights = learner.get_alphaweights()
        numRounds = np.sum(weights!=0,0)
        savearray(numRounds,'log/ABC2_T')

        start_time = time()
        learner = AdaBoostC2(base='svm')
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], 'ABC2_SVM', evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
        weights = learner.get_alphaweights()
        numRounds = np.sum(weights!=0,0)
        savearray(numRounds,'log/ABC2_T')