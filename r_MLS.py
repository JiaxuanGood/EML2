from u_base import *
from u_datastream import *
from a_MLDE import MLDE

if __name__ == '__main__':
    numdata = 14
    datasnames = ["Birds","CHD_49","Corel5k","Emotions","GpositiveGO","Image","Langlog","Scene","Chemistry","Philosophy","Tmc2007_500","Water_quality","Yeast","Yelp"]
    rd = ReadData(datas=datasnames,genpath='data/')

    num_trunk = 6
    '''k-fold with 1 result'''
    for dataIdx in range(numdata):
    # for dataIdx in (2,8,9,10):
        print(dataIdx)
        X,Y = rd.readDataall(dataIdx)
        mstream = stream(X,Y, num_trunk=num_trunk)
        X0, Y0 = mstream.datastream(0),mstream.datastream_label(0)

        start_time = time()
        learner = MLDE(5)
        learner.train(X0,Y0)
        mid_time = time()
        for trunk in range(num_trunk-1):
            Xtr = mstream.datastream(trunk+1)
            prediction = learner.test(Xtr)
            Ytr = mstream.datastream_label(trunk+1)
            saveResult(datasnames[dataIdx], 'MLDER'+str(trunk+1), evaluate(prediction, Ytr), (mid_time-start_time), (time()-mid_time))
            if(trunk==num_trunk-2):
                break
            start_time = time()
            learner.train(Xtr,Ytr)
            mid_time = time()
        
        start_time = time()
        learner = MLDE(5)
        learner.train(X0,Y0,stream=True)
        mid_time = time()
        for trunk in range(num_trunk-1):
            Xtr = mstream.datastream(trunk+1)
            prediction = learner.test(Xtr)
            Ytr = mstream.datastream_label(trunk+1)
            saveResult(datasnames[dataIdx], 'MLDEnR'+str(trunk+1), evaluate(prediction, Ytr), (mid_time-start_time), (time()-mid_time))
            if(trunk==num_trunk-2):
                break
            start_time = time()
            learner.train2(Xtr,Ytr)
            mid_time = time()
        
        start_time = time()
        learner = MLDE(10)
        learner.train(X0,Y0,stream=True)
        mid_time = time()
        for trunk in range(num_trunk-1):
            Xtr = mstream.datastream(trunk+1)
            prediction = learner.test(Xtr)
            Ytr = mstream.datastream_label(trunk+1)
            saveResult(datasnames[dataIdx], 'MLDE2R'+str(trunk+1), evaluate(prediction, Ytr), (mid_time-start_time), (time()-mid_time))
            if(trunk==num_trunk-2):
                break
            start_time = time()
            learner.train2(Xtr,Ytr,replace=True)
            mid_time = time()
            