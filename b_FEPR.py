from u_base import *
from w_weaksample import *
from w_weakcomplete import *
from w_prime import primes,getEntMat

'''additional BR classifier'''
def basing(X,Y,Xt):
    br = BR()
    br.train(X,Y)
    return br.test(Xt)

'''semi-completed label -> ranking classification label'''
def rankLabel(y1,y2):
    y_rank,idx_rank,y_weight = [],[],[]
    for i in range(len(y1)):
        # if(y1[i]!=y2[i]):
        if(np.abs(y1[i]-y2[i])>ranking_threshold): # ranking threshold
            this_weight = np.abs(y1[i]-y2[i])
            y_weight.append(this_weight)
            idx_rank.append(i)
            if(y1[i]>y2[i]):
                y_rank.append(1)
            else:
                y_rank.append(0)
    savearray([len(y_rank)],'log/HEPRtau'+str(ranking_threshold))
    if(noweight==True):
        y_weight = []
        # print('noweight')
    return y_rank,idx_rank,y_weight

'''full connected graph'''
def RANKf(X,Y_new,Xt,Yt,basis=None):
    M,Q = np.shape(Yt)
    prediction = np.zeros((Q,M))
    t1,t2 = 0,0
    for i in range(Q):
        for j in range(i+1,Q):
            t0 = time()
            y_rank,idx_rank,y_weight = rankLabel(Y_new[:,i],Y_new[:,j])
            thisLearner = Baser() #y_weight = y_weight/np.sum(y_weight) * len(y_weight)
            thisLearner.fit(X[idx_rank], y_rank, y_weight)
            t1 += time()-t0
            t0 = time()
            prd_rank = thisLearner.predict_proba(Xt)
            prd_rank = np.transpose(prd_rank)
            alpha = 2*np.abs(prd_rank[1]-0.5)
            if(basis is None):
                prediction[i] += prd_rank[1]
                prediction[j] += prd_rank[0]
            else:
                prediction[i] += alpha*prd_rank[1]+(1-alpha)*basis[:,i]
                prediction[j] += alpha*prd_rank[0]+(1-alpha)*basis[:,j]
            t2 += time()-t0
    prediction /= (Q-1)
    prediction = np.transpose(prediction)
    return prediction,t1,t2

'''a heuristic ranking system'''
def RANKa(X,Y_new,Xt,Yt,elist,basis=None):
    M,Q = np.shape(Yt)
    cnts = np.zeros(Q)
    tmp = np.array(elist).flatten().tolist()
    for i in range(len(tmp)):
        cnts[tmp[i]] += 1
    prediction = np.zeros((Q,M))
    t1,t2 = 0,0
    for r in range(Q-1):
        i=elist[r][0]
        j=elist[r][1]
        t0 = time()
        y_rank,idx_rank,y_weight = rankLabel(Y_new[:,i],Y_new[:,j])
        thisLearner = Baser()
        thisLearner.fit(X[idx_rank], y_rank, y_weight)
        t1 += time()-t0
        t0 = time()
        prd_rank = thisLearner.predict_proba(Xt)
        prd_rank = np.transpose(prd_rank)
        alpha = 2*np.abs(prd_rank[1]-0.5)
        if(basis is None):
            prediction[i] += prd_rank[1]
            prediction[j] += prd_rank[0]
        else:
            prediction[i] += alpha*prd_rank[1]+(1-alpha)*basis[:,i]
            prediction[j] += alpha*prd_rank[0]+(1-alpha)*basis[:,j]
        t2 += time()-t0
    for i in range(Q):
        prediction[i] /= cnts[i]
    prediction = np.transpose(prediction)
    return prediction,t1,t2

'''EARS: ranking with heuristically guided ranking systems'''
def RANK(X,Y_new,Xt,Yt,elists,basis=None):
    prediction = np.zeros(np.shape(Yt))
    t1,t2 = 0,0
    for i in range(len(elists)):
        tmp_prediction,tmp_t1,tmp_t2 = RANKa(X,Y_new,Xt,Yt,elists[i],basis)
        prediction += tmp_prediction
        t1 += tmp_t1
        t2 += tmp_t2
    return prediction/len(elists),t1,t2

def AEPR(X,Y_new,Xt,Yt,basis):
    num_label = np.shape(Y_new)[1]
    prediction,t1,t2 = RANKa(X,Y_new,Xt,Yt,primes(getEntMat(Y_new),random.sample(range(num_label),1))[0],basis)
    return prediction,t0+t1,t2

def HEPR(X,Y_new,Xt,Yt,basis=None):
    num_label = np.shape(Y_new)[1]
    num_tree = int(round(np.sqrt(num_label)))
    edgeBegins = random.sample(range(num_label),num_tree)
    prediction,t1,t2 = RANK(X,Y_new,Xt,Yt,primes(getEntMat(Y_new),edgeBegins),basis)
    return prediction,t0+t1,t2

def FEPR(X,Y_new,Xt,Yt,basis):
    prediction,t1,t2 = RANKf(X,Y_new,Xt,Yt,basis)
    return prediction,t1,t2

if __name__ == '__main__':
    numdata = 4
    datasnames = ["Image","Yeast","Langlog","Chemistry"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    kb = 20
    misrate = 0.3
    k_this = int(kb*misrate)
    noweight = False
    ranking_threshold = 0.5

    misrate = 0.3
    for dataIdx in range(numdata):
        for para_tau in range(5):
            ranking_threshold = (para_tau+1)*0.2
            X,Y,Xt,Yt = rd.readData(dataIdx)
            mode = 'mis'
            Y2 = mistaking(np.array(Y),misrate,mode)
            Y_new = completeLabel(X,Y2,k=k_this, complete_which=0, self_weight=1)
            t0 = time()
            basis = basing(X,intlabel(Y_new,0),Xt)
            t0 = time()-t0
            prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt,basis)
            saveResult(datasnames[dataIdx], 'HEPR_'+str(ranking_threshold), evaluate(prediction, Yt), traintime+t0, testtime)

    '''ablation2'''
    # for dataIdx in range(numdata):
    #     X,Y,Xt,Yt = rd.readData(dataIdx)
    #     mode = 'mis'
    #     Y2 = mistaking(np.array(Y),misrate,mode)
    #     Y_new = completeLabel(X,Y2,k=k_this, complete_which=0, self_weight=1)
    #     t0 = time()
    #     basis = basing(X,intlabel(Y_new,0),Xt)
    #     t0 = time()-t0
    #     noweight = False
    #     ranking_threshold = 0.5
    #     prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt,basis)
    #     saveResult(datasnames[dataIdx], 'HEPR', evaluate(prediction, Yt), traintime+t0, testtime)
    #     ranking_threshold = 0
    #     prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt,basis)
    #     saveResult(datasnames[dataIdx], 'Base4', evaluate(prediction, Yt), traintime+t0, testtime)
    #     noweight = True
    #     prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt,basis)
    #     saveResult(datasnames[dataIdx], 'Base3', evaluate(prediction, Yt), traintime+t0, testtime)
    #     prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt)
    #     saveResult(datasnames[dataIdx], 'Base2', evaluate(prediction, Yt), traintime+t0, testtime)
    #     prediction,traintime,testtime = HEPR(X,Y2,Xt,Yt)
    #     saveResult(datasnames[dataIdx], 'Base1', evaluate(prediction, Yt), traintime+t0, testtime)

    '''parameter k'''
    # misrate = 0.7
    # for para_k in range(1,2):
    #     k_this = (para_k+1)*4
    #     for dataIdx in range(3,numdata):
    #         X,Y,Xt,Yt = rd.readData(dataIdx)
    #         mode = 'mis'
    #         Y2 = mistaking(np.array(Y),misrate,mode)
    #         Y_new = completeLabel(X,Y2,k=k_this, complete_which=0, self_weight=1)
    #         t0 = time()
    #         basis = basing(X,intlabel(Y_new,0),Xt)
    #         t0 = time()-t0
    #         prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt,basis)
    #         saveResult(datasnames[dataIdx], 'HEPR_'+str(k_this), evaluate(prediction, Yt), traintime+t0, testtime)

    '''ablation1'''
    # for dataIdx in range(numdata):
    #     X,Y,Xt,Yt = rd.readData(dataIdx)
    #     mode = 'mis'
    #     Y2 = mistaking(np.array(Y),misrate,mode)
    #     Y_new = completeLabel(X,Y2,k=k_this, complete_which=0, self_weight=1)
    #     t0 = time()
    #     basis = basing(X,intlabel(Y_new,0),Xt)
    #     t0 = time()-t0
    #     prediction,traintime,testtime = AEPR(X,Y_new,Xt,Yt,basis)
    #     saveResult(datasnames[dataIdx], 'FEPR', evaluate(prediction, Yt), traintime+t0, testtime)
    #     prediction,traintime,testtime = HEPR(X,Y_new,Xt,Yt,basis)
    #     saveResult(datasnames[dataIdx], 'HEPR', evaluate(prediction, Yt), traintime+t0, testtime)
    #     prediction,traintime,testtime = FEPR(X,Y_new,Xt,Yt,basis)
    #     saveResult(datasnames[dataIdx], 'AEPR', evaluate(prediction, Yt), traintime+t0, testtime)
        
