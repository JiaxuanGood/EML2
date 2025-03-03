from u_base import *
from s_feature import *
from s_utils import *
from s_optimizer import optimizer as metalearner

class SCTM():
    def __init__(self, tau=0.7, para_opt=(0.1,0.01,0.01,200,0.0001), numcluster=8, duplic=2):
        self.tau0 = tau
        self.alpha = para_opt[0]
        self.beta = para_opt[1]
        self.enta = para_opt[2]
        self.maxItr = para_opt[3]
        self.tol = para_opt[4]
        self.numclus = numcluster
        self.learners = []
        if(duplic==1 or duplic==2 or duplic==3): # duplic=2,1 for base learners on Xl will be finally considered, or not 
            self.duplic = duplic
        else:
            self.duplic = 2
    
    def train(self, Xl, Yl, Xu):
        numbase = 3
        self.numbase = numbase
        numlabel = np.shape(Yl)[1]
        X = np.vstack((Xl,Xu))
        self.idxset = getidxset_dask(X,Yl,numbase)
        Xls = getsubX(Xl, self.idxset)
        Xus = getsubX(Xu, self.idxset)
        
        '''Learner Generation'''
        baseLearners = []
        for i in range(numbase):
            br = CC()
            br.train(Xls[i],Yl)
            baseLearners.append(br)
            if(self.duplic>1):
                self.learners.append(br)
        '''Data Expansion'''
        locfit = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        fakelabel = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        for i in range(numbase):#numbase
            fakeresult = baseLearners[i].test(Xus[i])
            for j in range(numlabel):
                tau = self.tau0
                fakelabel[i][j] = np.round(fakeresult[:,j])
                prob = np.argmax(np.array([fakeresult[:,j],1-fakeresult[:,j]]), axis=0)
                locfit[i][j] = np.array(prob>tau)*1
        '''self-training learners'''
        for i in range(numbase):
            tr_idx = np.array(locfit[i])
            thislearner = CC()
            thislearner.train(np.vstack((Xls[i],Xus[i])), np.vstack((Yl,np.transpose(np.array(fakelabel[i])))), tr_idx)
            self.learners.append(thislearner)
        '''Interactive annotation'''
        locfit2 = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        for i in range(numbase):#numbase
            p = (i+1)%numbase
            q = (i+2)%numbase
            for j in range(numlabel):
                locfit2[i][j] = np.array(fakelabel[p][j]==fakelabel[q][j])*1
        '''Learner Enhance'''
        for i in range(numbase):
            tr_idx = np.array(locfit[i])*np.array(locfit2[i]) # (numLabel,numData)
            thislearner = CC()
            thislearner.train(np.vstack((Xls[i],Xus[i])), np.vstack((Yl,np.transpose(np.array(fakelabel[i])))), tr_idx)
            self.learners.append(thislearner)

        self.Xls = Xls
        self.Xus = Xus
    
    def trainmeta(self, Xl, Xu, Yl, base_idx):
        print(base_idx)
        self.base_idx = base_idx
        '''meta learner'''
        new_Xl = []
        new_Xu = []
        for i in base_idx:
            new_Xl.append(self.learners[i].test(self.Xls[i%self.numbase]))
            new_Xu.append(self.learners[i].test(self.Xus[i%self.numbase]))
        new_Xl = np.hstack(tuple(new_Xl))
        new_Xu = np.hstack(tuple(new_Xu))
        '''
        alpha: L1
        beta: manifold
        enta: label correlation
        '''
        self.W = metalearner(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, self.alpha, self.beta, self.enta, self.maxItr, self.tol, self.numclus) # SCTML

    def test(self, Xt):
        Xts = getsubX(Xt, self.idxset)
        new_Xt = []
        for i in self.base_idx:
            new_Xt.append(self.learners[i].test(Xts[i%self.numbase]))
        new_Xt = np.hstack(tuple(new_Xt))
        Pt = np.dot(new_Xt, self.W)
        return Pt

if __name__ == '__main__':
    numdata = 14
    # datasnames = ["Birds","CAL500","CHD_49","Corel5k","Emotions","Enron","Flags","Foodtruck","Genbase","GnegativeGO",
    #     "GpositiveGO","HumanGO","Image","Langlog","Mediamill","Medical","Ohsumed","PlantGO","Scene","Slashdot",
    #     "Chemistry","Chess","Coffee","Philosophy","Tmc2007_500","VirusGO","Water_quality","Yeast","Yelp"]
    # datasnames = ["Bibtex","Delicious","EukaryoteGO","Imdb","Cooking","CS","Arts","Business","Entertainment","Health","Recreation","Science"]
    # datasnames = ["Bookmarks","Tmc2007","Computers","Education","Reference","Social","Society"]
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","Genbase","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')

    '''k-fold with 1 result'''
    r_unlabeled = [1,4,9]
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for z in r_unlabeled:
            Xl,Yl,Xu,Yu = datasplit(X,Y, 1/(1+z))

            start_time = time()
            learner = SCTM()
            learner.train(Xl,Yl,Xu)
            learner.trainmeta(Xl,Xu,Yl,np.arange(9))
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_9_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            learner.trainmeta(Xl,Xu,Yl,np.arange(3)+3)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_3s_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            learner.trainmeta(Xl,Xu,Yl,np.arange(3)+6)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_3i_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            learner.trainmeta(Xl,Xu,Yl,np.arange(6))
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_6s_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            learner.trainmeta(Xl,Xu,Yl,[0,1,2,6,7,8])
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_6i_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            
