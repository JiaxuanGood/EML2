from u_base import *
from s_feature import *
from s_utils import *
from s_Meta_Base import meta1
from s_Meta_Cluster import meta2
from s_Meta_Cluster_w import meta3
from s_Meta_dLap import meta4
from a_SCTM import SCTM
from s_optimizer import optimizer as metalearner

class SCTMmeta(SCTM):
    def trainmeta(self, Xl, Xu, Yl, new_Xl, new_Xu):
        self.W = metalearner(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, self.alpha, self.beta, self.enta, self.maxItr, self.tol, self.numclus) # SCTML
        self.Xa = np.vstack((Xl,Xu))
        self.Yl, self.new_Xl, self.new_Xu = Yl, new_Xl, new_Xu
    def trainmeta2(self, switch):
        if(switch==0):
            tmp = [np.eye(np.shape(self.Yl)[1])] * (self.numbase*self.duplic)
            self.W = np.vstack(tuple(tmp)) / (self.numbase*self.duplic)
        if(switch==1):
            self.W = meta1(self.Xa, self.new_Xu, self.new_Xl, self.Yl, self.alpha, self.beta, self.maxItr, self.tol) # stacking only
        if(switch==4):
            self.W = meta4(self.Xa, self.new_Xu, self.new_Xl, self.Yl, self.alpha, self.beta, self.enta, self.maxItr, self.tol) # stacking withnot clustering
        if(switch==2):
            self.W = meta2(self.Xa, self.new_Xu, self.new_Xl, self.Yl, self.alpha, self.beta, self.maxItr, self.tol, self.numclus) # stacking without label correlation, with clustering
        if(switch==3):
            self.W = meta3(self.Xa, self.new_Xu, self.new_Xl, self.Yl, self.alpha, self.beta, self.maxItr, self.tol, self.numclus) # stacking without label correlation, with weighted clustering

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

            learner = SCTMmeta()
            learner.train(Xl,Yl,Xu)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM_'+str(z), evaluate(prediction, Yt))

            learner.trainmeta2(0)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM0_'+str(z), evaluate(prediction, Yt))
            learner.trainmeta2(1)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM1_'+str(z), evaluate(prediction, Yt))
            learner.trainmeta2(2)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM2_'+str(z), evaluate(prediction, Yt))
            learner.trainmeta2(3)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM3_'+str(z), evaluate(prediction, Yt))
            learner.trainmeta2(4)
            prediction = learner.test(Xt)
            saveResult(datasnames[dataIdx], 'SCTM4_'+str(z), evaluate(prediction, Yt))
