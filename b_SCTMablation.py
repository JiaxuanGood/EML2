from u_base import *
from s_feature import *
from s_utils import *
from s_optimizer import optimizer as metalearner
from b_SCTMMetas import meta,meta1,meta2,meta3,meta4

if __name__ == '__main__':
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    duplic = 2
    tau = 0.7

    '''k-fold with 1 result'''
    r_unlabeled = [1,4]
    for dataIdx in range(14):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for z in r_unlabeled:
            Xl,Yl,Xu,Yu = datasplit(X,Y, 1/(1+z))
            

            START_TIME = time()
            numbase = 3
            numlabel = np.shape(Yl)[1]
            X = np.vstack((Xl,Xu))
            idxset = getidxset_rand(X,numbase)
            # idxset = getidxset_dask(X,Yl,numbase)
            Xls = getsubX(Xl, idxset)
            Xus = getsubX(Xu, idxset)
            
            learners = []
            '''Learner Generation'''
            baseLearners = []
            for i in range(numbase):
                cc = CC()
                cc.train(Xls[i],Yl)
                baseLearners.append(cc)
                if(duplic>1):
                    learners.append(cc)
            '''Data Expansion'''
            locfit = [[[] for _ in range(numlabel)] for _ in range(numbase)]
            fakelabel = [[[] for _ in range(numlabel)] for _ in range(numbase)]
            for i in range(numbase):#numbase
                fakeresult = np.transpose(baseLearners[i].test(Xus[i]))
                fakelabel[i] = np.round(fakeresult)
                locfit[i] = np.array(fakeresult>tau)*1+np.array(fakeresult<1-tau)*1
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
                # this_data = np.vstack((Xls[i],Xus[i]))
                # this_target = np.vstack((Yl,np.transpose(fakelabel[i])))
                # tr_idx = np.transpose(np.vstack((np.ones(np.shape(Yl)),np.transpose(fakelabel[i]))))
                # br.train(this_data, this_target, tr_idx)
                br.train(Xus[i], np.transpose(fakelabel[i]), tr_idx)
                learners.append(br)
            '''meta learner'''
            new_Xl = []
            new_Xu = []
            for i in range(numbase*duplic):
                new_Xl.append(learners[i].test(Xls[i%numbase]))
                new_Xu.append(learners[i].test(Xus[i%numbase]))
            new_Xl = np.hstack(tuple(new_Xl))
            new_Xu = np.hstack(tuple(new_Xu))
            TRAIN_TIME = time()-START_TIME

            Xts = getsubX(Xt, idxset)
            new_Xt = []
            for i in range(numbase*duplic):
                new_Xt.append(learners[i].test(Xts[i%numbase]))
            new_Xt = np.hstack(tuple(new_Xt))
            '''
            alpha: L1
            beta: manifold
            enta: label correlation
            '''
            MID_TIME = time()
            tmp = [np.eye(np.shape(Yl)[1]) for _ in range(numbase*duplic)]
            tmp = tuple(tmp)
            W = np.vstack(tmp) / 3 # ECC+CT
            Pt = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'AVG', evaluate(Pt, Yt), TRAIN_TIME, (time()-MID_TIME))

            MID_TIME = time()
            W = meta1(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, 0.1, 0.01, 200, 0.0001) # stacking only
            Pt = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'BASE', evaluate(Pt, Yt), TRAIN_TIME, (time()-MID_TIME))

            MID_TIME = time()
            W = meta4(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, 0.1, 0.01, 0.01, 200, 0.0001) # stacking withnot clustering
            Pt = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'manifold2', evaluate(Pt, Yt), TRAIN_TIME, (time()-MID_TIME))

            MID_TIME = time()
            W = meta2(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, 0.1, 0.01, 200, 0.0001, 8) # stacking without label correlation, with clustering
            Pt = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'cluster', evaluate(Pt, Yt), TRAIN_TIME, (time()-MID_TIME))

            MID_TIME = time()
            W = meta3(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, 0.1, 0.01, 200, 0.0001, 8) # stacking without label correlation, with weighted clustering
            Pt = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'cluster_w', evaluate(Pt, Yt), TRAIN_TIME, (time()-MID_TIME))

            MID_TIME = time()
            W = meta(np.vstack((Xl,Xu)), new_Xu, new_Xl, Yl, 0.1, 0.01, 0.01, 200, 0.0001, 8) # SCTML
            Pt = np.dot(new_Xt, W)
            saveResult(datasnames[dataIdx], 'SCTML', evaluate(Pt, Yt), TRAIN_TIME, (time()-MID_TIME))
            