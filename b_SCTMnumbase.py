from u_base import *
from s_feature import *
from s_utils import *
from s_optimizer import optimizer as metalearner

class SCTMbase():
    def __init__(self, numbase, tau=0.7, para_opt=(0.1,0.01,0.01,200,0.0001), numcluster=8, duplic=2):
        self.numbase = numbase
        self.tau0 = tau
        self.alpha = para_opt[0]
        self.beta = para_opt[1]
        self.enta = para_opt[2]
        self.maxItr = para_opt[3]
        self.error = para_opt[4]
        self.numclus = numcluster
        self.learners = []
        if(duplic==1 or duplic==2): # duplic=2,1 for base learners on Xl will be finally considered, or not 
            self.duplic = duplic
        else:
            self.duplic = 2
    
    def train(self, Xl, Yl, Xu):
        numbase = self.numbase
        numlabel = np.shape(Yl)[1]
        X = np.vstack((Xl,Xu))
        self.idxset = getidxset_dask(X,Yl,numbase)
        Xls = getsubX(Xl, self.idxset)
        Xus = getsubX(Xu, self.idxset)
        
        '''Learner Generation'''
        baseLearners = []
        for i in range(numbase):
            br = BR()
            br.train(Xls[i],Yl)
            baseLearners.append(br)
            if(self.duplic==2):
                self.learners.append(br)
        '''Data Expansion'''
        locfit = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        fakelabel = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        for i in range(numbase):#numbase
            for j in range(numlabel):
                tau = self.tau0
                tmp = baseLearners[i].test_a(Xus[i],j)
                fakelabel[i][j] = np.round(tmp)
                tmp = np.abs(tmp*2-1)
                locfit[i][j] = np.array(tmp>tau)*1
                if(sum(locfit[i][j])<len(Xl)*(1-self.tau0)):
                    tau = np.sort(tmp)[int(self.tau0*len(tmp))]
                    locfit[i][j] = np.array(tmp>tau)*1
        self.numlabel = numlabel
        self.locfit = locfit
        self.fakelabel = fakelabel
        self.orglen = len(Xu)
        self.Xls,self.Xus,self.Yl = Xls,Xus,Yl
    
    def getloc2(self):
        '''Interactive annotation'''
        locfit2 = np.ones((self.numbase,self.numlabel,self.orglen))
        self.locfit2 = locfit2
        self.trainCC()

    def getloc3(self):
        locfit2 = [[[] for _ in range(self.numlabel)] for _ in range(self.numbase)]
        for i in range(self.numbase):#numbase
            p = (i+1)%self.numbase
            q = (i+2)%self.numbase
            for j in range(self.numlabel):
                locfit2[i][j] = np.array(self.fakelabel[p][j]==self.fakelabel[q][j])*1
        self.locfit2 = locfit2
        self.trainCC()

    def getloc4(self):
        locfit2 = [[[] for _ in range(self.numlabel)] for _ in range(self.numbase)]
        for i in range(self.numbase):#numbase
            p = (i+1)%self.numbase
            q = (i+2)%self.numbase
            zp = (i+3)%self.numbase
            for j in range(self.numlabel):
                locfit2[i][j] = np.array(self.fakelabel[p][j]==self.fakelabel[q][j])*np.array(self.fakelabel[p][j]==self.fakelabel[zp][j])*1
        self.locfit2 = locfit2
        self.trainCC()

    def getloc5(self):
        locfit2 = [[[] for _ in range(self.numlabel)] for _ in range(self.numbase)]
        for i in range(self.numbase):#numbase
            p = (i+1)%self.numbase
            q = (i+2)%self.numbase
            zp = (i+3)%self.numbase
            zq = (i+4)%self.numbase
            for j in range(self.numlabel):
                locfit2[i][j] = np.array(self.fakelabel[p][j]==self.fakelabel[q][j])*np.array(self.fakelabel[p][j]==self.fakelabel[zp][j])*np.array(self.fakelabel[p][j]==self.fakelabel[zq][j])*1
        self.locfit2 = locfit2
        self.trainCC()

    def getloc6(self):
        locfit2 = [[[] for _ in range(self.numlabel)] for _ in range(self.numbase)]
        for i in range(self.numbase):#numbase
            p = (i+1)%self.numbase
            q = (i+2)%self.numbase
            zp = (i+3)%self.numbase
            zq = (i+4)%self.numbase
            zzz = (i+5)%self.numbase
            for j in range(self.numlabel):
                locfit2[i][j] = np.array(self.fakelabel[p][j]==self.fakelabel[q][j])*np.array(self.fakelabel[p][j]==self.fakelabel[zp][j])*np.array(self.fakelabel[p][j]==self.fakelabel[zq][j])*np.array(self.fakelabel[p][j]==self.fakelabel[zzz][j])*1
        self.locfit2 = locfit2
        self.trainCC()

    def trainCC(self):
        newlearner = []
        for i in range(self.numbase):
            newlearner.append(self.learners[i])
        '''Learner Enhance'''
        for i in range(self.numbase):
            tr_idx = np.array(self.locfit[i])*np.array(self.locfit2[i]) # (numLabel,numData)
            thislearner = CC()
            thislearner.train(np.vstack((self.Xls[i],self.Xus[i])), np.vstack((self.Yl,np.transpose(np.array(self.fakelabel[i])))), tr_idx)
            newlearner.append(thislearner)
        self.learners = newlearner
        print(len(self.learners))
        new_Xl = []
        new_Xu = []
        for i in range(self.numbase*self.duplic):
            new_Xl.append(self.learners[i].test(self.Xls[i%self.numbase]))
            new_Xu.append(self.learners[i].test(self.Xus[i%self.numbase]))
        new_Xl = np.hstack(tuple(new_Xl))
        new_Xu = np.hstack(tuple(new_Xu))
        '''
        alpha: L1
        beta: manifold
        enta: label correlation
        '''
        self.W = metalearner(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, self.alpha, self.beta, self.enta, self.maxItr, self.error, self.numclus) # SCTML

    def test(self, Xt):
        Xts = getsubX(Xt, self.idxset)
        new_Xt = []
        for i in range(self.numbase*self.duplic):
            new_Xt.append(self.learners[i].test(Xts[i%self.numbase]))
        new_Xt = np.hstack(tuple(new_Xt))
        Pt = np.dot(new_Xt, self.W)
        return Pt

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Birds","CAL500","CHD_49","Corel5k","Emotions","Enron","Flags","Foodtruck","Genbase","GnegativeGO",
        "GpositiveGO","HumanGO","Image","Langlog","Mediamill","Medical","Ohsumed","PlantGO","Scene","Slashdot",
        "Chemistry","Chess","Coffee","Philosophy","Tmc2007_500","VirusGO","Water_quality","Yeast","Yelp"]
    # datasnames = ["Bibtex","Delicious","EukaryoteGO","Imdb","Cooking","CS","Arts","Business","Entertainment","Health","Recreation","Science"]
    # datasnames = ["Bookmarks","Tmc2007","Computers","Education","Reference","Social","Society"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')

    '''k-fold with 1 result'''
    n_fold = 10
    r_labeled = [1,3]
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for z in range(2):
            Xl,Yl,Xu,Yu = datasplit(X,Y, r_labeled[z]/(n_fold-1))

            start_time = time()
            learner = SCTMbase(2)
            learner.train(Xl,Yl,Xu)
            learner.getloc2()
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTMb2_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            start_time = time()
            learner = SCTMbase(3)
            learner.train(Xl,Yl,Xu)
            learner.getloc2()
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTMb3_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            start_time = time()
            learner = SCTMbase(4)
            learner.train(Xl,Yl,Xu)
            learner.getloc2()
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTMb4_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            start_time = time()
            learner = SCTMbase(5)
            learner.train(Xl,Yl,Xu)
            learner.getloc2()
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTMb5_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

            start_time = time()
            learner = SCTMbase(6)
            learner.train(Xl,Yl,Xu)
            learner.getloc2()
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTMb6_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

