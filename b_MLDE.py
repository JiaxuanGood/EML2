from u_base import *
from sklearn.neighbors import NearestNeighbors

class MLDE():
    def __init__(self, numbase=30, numneibor=10):
        self.numbase = numbase
        self.numneibor = numneibor
        self.base = []
        self.scores = []
    
    def train(self, traX, traY, ratio=0.8):
        self.findNb = NearestNeighbors(n_neighbors=self.numneibor, algorithm='ball_tree')
        self.findNb.fit(traX)
        numdata,self.numlabel = np.shape(traY)
        for i in range(self.numbase):
            br = BR()
            sampledId = random.sample(range(0,numdata),int(ratio*numdata))
            br.train(traX[sampledId], traY[sampledId])
            self.base.append(br)
    
    def train_score_SL(self,traX,traY):
        scores = np.zeros((self.numbase,numdata,self.numlabel))
        for i in range(self.numbase):
            br = self.base.append[i]
            output = br.test(traX)
            for j in range(len(traX)):
                for k in range(self.numlabel):
                    scores[i,j,k] = int(np.round(output[j,k])==traY[j,k])
        self.scores = scores

    def train_score_ML(self,traX,traY,metricid=6): # 6 for Hamming loss, 9 for Ranking loss
        scores = np.zeros((self.numbase,numdata))
        for i in range(self.numbase):
            br = self.base.append[i]
            output = br.test(traX)
            for j in range(len(traX)):
                measure = evaluate([output[j]], [traY[j]])
                scores[i,j] = measure[metricid]
        self.scores = scores

    def test_SL(self, tesX):
        scores = self.scores
        distances,indices = self.findNb.kneighbors(tesX)
        idx = []
        for i in range(len(tesX)):
            s1_i = np.zeros((self.numbase,self.numlabel))
            for j in range(self.numbase):
                for nb in indices[i]:
                    s1_i[j] += scores[j,nb,:]
            tmp = []
            for k in range(self.numlabel):
                tmp.append(np.argwhere(s1_i[:,k]==np.min(s1_i[:,k])).flatten())
            idx.append(tmp)
        prediction = np.zeros((len(tesX),self.numlabel))
        for i in range(len(tesX)):
            for k in range(self.numlabel):
                for j in idx[i][k]:
                    prediction[i,k] += self.base[j].test_a([tesX[i]],k)[0]
                prediction[i,k] /= len(idx[i][k])
        return prediction

    def test_ML(self, tesX):
        scores = self.scores
        distances,indices = self.findNb.kneighbors(tesX)
        idx = []
        for i in range(len(tesX)):
            s_i = np.zeros(self.numbase)
            for j in range(self.numbase):
                for nb in indices[i]:
                    s_i[j] += scores[j,nb]
            idx.append(np.argsort(s_i)[:int(self.numneibor/2)])
        prediction = np.zeros((len(tesX),self.numlabel))
        for i in range(len(tesX)):
            for j in idx[i]:
                prediction[i] += self.base[j].test([tesX[i]])[0]
            prediction[i] /= len(idx[i])
        return prediction

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'MLDE'

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = MLDE()
        learner.train(X,Y)
        learner.train_score_SL(X,Y)
        learner.train_score_ML(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
