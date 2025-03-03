from u_base import *
from sklearn.neighbors import NearestNeighbors

class MLDE():
    def __init__(self, numbase=10, numneibor=10):
        self.numbase = numbase
        self.numneibor = numneibor
        self.base = []
    
    def train(self, traX, traY, ratio=0.8, stream=False):
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
        if(stream):
            self.traX = traX
            self.traY = traY
    
    def train2(self, Xnew, Ynew, ratio=0.8, replace=False):
        if(replace):
            loss = []
            for br in self.base:
                output = br.test(Xnew)
                scores = evaluate(output, Ynew)
                loss.append(scores[6]+scores[9])
            loss = 1-np.array(loss)
            median = np.percentile(loss, 50)
            newbase = []
            for i in range(self.numbase):
                if(loss[i]<median):
                    newbase.append(self.base[i])
            self.numbase = int(self.numbase/2)
            self.base = newbase
            print('numbase',self.numbase,len(self.base))
        for i in range(self.numbase):
            br = BR()
            sampledId = random.sample(range(0,len(Xnew)),int(ratio*len(Xnew)))
            br.train(Xnew[sampledId], Ynew[sampledId])
            self.base.append(br)
        self.numbase = len(self.base)
        self.traX = np.vstack((self.traX, Xnew))
        self.traY = np.vstack((self.traY, Ynew))
        self.findNb = NearestNeighbors(n_neighbors=self.numneibor, algorithm='ball_tree')
        self.findNb.fit(self.traX)
        numdata = len(self.traX)
        scores1 = np.zeros((self.numbase,numdata,self.numlabel))
        scores2 = np.zeros((self.numbase,numdata))
        for i in range(self.numbase):
            br = self.base[i]
            output = br.test(self.traX)
            for j in range(len(self.traX)):
                for k in range(self.numlabel):
                    scores1[i,j,k] = int(np.round(output[j,k])==self.traY[j,k])
                measure = evaluate([output[j]], [self.traY[j]])
                scores2[i,j] = measure[9]
        self.scores = [scores1, scores2]

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
            # saveMat(s1_i,2)
            # saveMat(s2_i,1)
            for k in range(self.numlabel):
                tmp.append(np.argwhere(s1_i[:,k]==np.max(s1_i[:,k])).flatten())
            idx1.append(tmp)
            idx2.append(np.argsort(s2_i)[:int(self.numbase/2)])
        prediction1 = np.zeros((len(tesX),self.numlabel))
        prediction2 = np.zeros((len(tesX),self.numlabel))
        # saveMat(idx1,3)
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
    numdata = 1
    datasnames = ["Yeast"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'MLDE'
    numbase = 5

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = MLDE(numbase)
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))
