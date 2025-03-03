from u_base import *
from sklearn.neighbors import NearestNeighbors

class MLDE():
    def __init__(self, numbase=30, numneibor=10):
        self.numbase = numbase
        self.numneibor = numneibor
        self.base = []

    def setnumbase(self, numbase):
        self.numbase = numbase
    
    def train(self, traX, traY, ratio=0.8):
        self.findNb = NearestNeighbors(n_neighbors=self.numneibor, algorithm='ball_tree')
        self.findNb.fit(traX)
        numdata,self.numlabel = np.shape(traY)
        scores1 = np.zeros((self.numbase,numdata,self.numlabel))
        scores2 = np.zeros((self.numbase,numdata))
        for i in range(self.numbase):
            br = BR()
            sampledId = random.sample(range(0,numdata),int(ratio*numdata))
            br.train(traX[sampledId], traY[sampledId])
            self.base.append(br)
            output = br.test(traX)
            for j in range(len(traX)):
                for k in range(self.numlabel):
                    scores1[i,j,k] = int(np.round(output[j,k])==traY[j,k])
                measure = evaluate([output[j]], [traY[j]])
                scores2[i,j] = measure[9]
        self.scores = [scores1, scores2]
        # saveMat(scores1,3)

    def test(self, tesX):
        scores1, scores2 = self.scores
        distances,indices = self.findNb.kneighbors(tesX)
        idx1,idx2 = [],[]
        for i in range(len(tesX)):
            s1_i = np.zeros((self.numbase,self.numlabel))
            s2_i = np.zeros(self.numbase)
            for j in range(self.numbase):
                for nb in indices[i]:
                    s1_i[j] += scores1[j,nb,:]
                    s2_i[j] += scores2[j,nb]
            tmp = []
            for k in range(self.numlabel):
                tmp.append(np.argwhere(s1_i[:,k]==np.max(s1_i[:,k])).flatten())
            idx1.append(tmp)
            idx2.append(np.argsort(s2_i)[:int(self.numbase/2)])
        prediction1 = np.zeros((len(tesX),self.numlabel))
        prediction2 = np.zeros((len(tesX),self.numlabel))
        for i in range(len(tesX)):
            for k in range(self.numlabel):
                for j in idx1[i][k]:
                    prediction1[i,k] += self.base[j].test_a([tesX[i]],k)[0]
                prediction1[i,k] /= len(idx1[i][k])
            for j in idx2[i]:
                prediction2[i] += self.base[j].test([tesX[i]])[0]
            prediction2[i] /= len(idx2[i])
        output = prediction1+prediction2
        return output/2
        
if __name__ == '__main__':
    numdata = 4
    datasnames = ["Image","Yeast","Langlog","Chemistry"]
    # rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'MLDE'

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = MLDE(16)
        learner.train(X,Y)
        mid_time = time()
        # for numbase in (5,10,15,20,25,30,35,40):
        for numbase in (6,8,10,12,14,16):
            t2 = time()
            learner.setnumbase(numbase)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], algname+str(numbase), evaluate(prediction, Yt), (mid_time-start_time), (time()-t2))
    