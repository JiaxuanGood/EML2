from u_base import *
from a_HEPR import HEPR_mis,HEPR_part,mistaking

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'HEPR_'
    kb = 20
    misrate = 0.3
    k_this = int(kb*misrate)
    for dataIdx in range(numdata):
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for mode in ('mis','mis7','part'):
            if(mode=='mis7'):
                Y2 = mistaking(np.array(Y),0.7,'mis')
            else:
                Y2 = mistaking(np.array(Y),misrate,mode)
            if(mode=='mis' or mode=='mis7'):
                prediction,traintime,testtime = HEPR_mis(X,Y2,Xt,Yt,k_this)
            if(mode=='part'):
                prediction,traintime,testtime = HEPR_part(X,Y2,Xt,Yt,k_this)
            saveResult(datasnames[dataIdx], algname+mode, evaluate(prediction, Yt), traintime, testtime)
    
    # for z in range(10):
    #     misrate = float(z/10)
    #     print(misrate)
    #     k_this = max(int(kb*misrate),8)
    #     for dataIdx in range(numdata):
    #         X,Y,Xt,Yt = rd.readData(dataIdx)
    #         Y2 = mistaking(np.array(Y),misrate,'mis')
    #         prediction,traintime,testtime = HEPR_mis(X,Y2,Xt,Yt,k_this)
    #         saveResult(datasnames[dataIdx], algname+'mis'+str(z), evaluate(prediction, Yt), traintime, testtime)
