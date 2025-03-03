from u_base import *
from s_feature import *
from s_utils import *
from s_optimizer import optimizer as metalearner

class SCTM():
    def __init__(self, tau=[0.7,0.7], para_opt=(0.1,0.01,0.01,200,0.0001), numcluster=8, duplic=2):
        self.tau = tau
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
        self.idxset = getidxset_rand(X,numbase)
        # self.idxset = getidxset_dask(X,Yl,numbase)
        Xls = getsubX(Xl, self.idxset)
        Xus = getsubX(Xu, self.idxset)
        
        '''Learner Generation'''
        baseLearners = []
        for i in range(numbase):
            cc = CC()
            cc.train(Xls[i],Yl)
            baseLearners.append(cc)
            if(self.duplic>1):
                self.learners.append(cc)
        '''Data Expansion'''
        locfit = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        fakelabel = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        for i in range(numbase):#numbase
            fakeresult = np.transpose(baseLearners[i].test(Xus[i]))
            fakelabel[i] = np.round(fakeresult)
            locfit[i] = np.array(fakeresult>self.tau[1])*1+np.array(fakeresult<1-self.tau[0])*1
        '''Interactive annotation'''
        locfit2 = [[[] for _ in range(numlabel)] for _ in range(numbase)]
        for i in range(numbase):#numbase
            p = (i+1)%numbase
            q = (i+2)%numbase
            locfit2[i] = np.array(fakelabel[p]==fakelabel[q])*1
        '''Learner Enhance'''
        for i in range(numbase):
            tr_idx = np.array(locfit[i])*np.array(locfit2[i]) # (numLabel,numData)
            br = BR()
            '''label + fake label'''
            # this_data = np.vstack((Xls[i],Xus[i]))
            # this_target = np.vstack((Yl,np.transpose(fakelabel[i])))
            # tr_idx = np.transpose(np.vstack((np.ones(np.shape(Yl)),np.transpose(fakelabel[i]))))
            # br.train(this_data, this_target, tr_idx)
            '''fake label'''
            br.train(Xus[i], np.transpose(fakelabel[i]), tr_idx)
            '''label'''
            # br.train(Xus[i], Yu, tr_idx)
            self.learners.append(br)
        '''meta learner'''
        new_Xl = []
        new_Xu = []
        for i in range(numbase*self.duplic):
            new_Xl.append(self.learners[i].test(Xls[i%numbase]))
            new_Xu.append(self.learners[i].test(Xus[i%numbase]))
        new_Xl = np.hstack(tuple(new_Xl))
        new_Xu = np.hstack(tuple(new_Xu))
        self.trainmeta(Xl, Xu, Yl, new_Xl, new_Xu)
    
    def trainmeta(self, Xl, Xu, Yl, new_Xl, new_Xu):
        '''
        alpha: L1
        beta: manifold
        enta: label correlation
        '''
        self.W = metalearner(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, self.alpha, self.beta, self.enta, self.maxItr, self.tol, self.numclus) # SCTML

    def test(self, Xt):
        Xts = getsubX(Xt, self.idxset)
        new_Xt = []
        for i in range(self.numbase*self.duplic):
            new_Xt.append(self.learners[i].test(Xts[i%self.numbase]))
        new_Xt = np.hstack(tuple(new_Xt))
        Pt = np.dot(new_Xt, self.W)
        return Pt

if __name__ == '__main__':
    numdata = 14
    # datasnames = ["Image","Yeast","Langlog","Chemistry"]
    # datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Scene","Philosophy","Tmc2007_500","Water_quality","Yelp"]
    
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'SCTM_'

    '''k-fold with 1 result'''
    r_unlabeled = [1,4]
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for z in r_unlabeled:
            Xl,Yl,Xu,Yu = datasplit(X,Y, 1/(1+z))
            start_time = time()
            learner = SCTM()
            learner.train(Xl,Yl,Xu)
            mid_time = time()
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_'+str(z), evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))

