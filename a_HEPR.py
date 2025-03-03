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
        if(np.abs(y1[i]-y2[i])>0.5): # ranking threshold
            this_weight = np.abs(y1[i]-y2[i])
            y_weight.append(this_weight)
            idx_rank.append(i)
            if(y1[i]>y2[i]):
                y_rank.append(1)
            else:
                y_rank.append(0)
    # winners = np.argwhere(y_weight > quantile(y_weight))
    # winners = winners.flatten()
    # y_rank = np.array(y_rank).take(winners)
    # idx_rank = np.array(idx_rank).take(winners)
    # y_weight = np.array(y_weight).take(winners)
    # print(len(y_rank), np.average(y_weight))
    savearray([len(y_rank)],'log/HEPRn')
    return y_rank,idx_rank,y_weight

'''full connected graph'''
def RANKf(X,Y_new,Xt,Yt,basis=np.zeros((1,1))):
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
            if(np.shape(basis)==(1,1)):
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
def RANKa(X,Y_new,Xt,Yt,elist,basis=np.zeros((1,1))):
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
        if(np.shape(basis)==(1,1)):
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

def HEPR(X,Y_new,Xt,Yt):
    num_label = np.shape(Y_new)[1]
    t0 = time()
    basis = basing(X,intlabel(Y_new,0),Xt)
    t0 = time()-t0
    num_tree = int(round(np.sqrt(num_label)))
    # print(num_tree)
    edgeBegins = random.sample(range(num_label),num_tree)
    prediction,t1,t2 = RANK(X,Y_new,Xt,Yt,primes(getEntMat(Y_new),edgeBegins),basis)
    # prediction,t1,t2 = RANKf(X,Y_new,Xt,Yt,basis)
    return prediction,t0+t1,t2

def HEPR_mis(X,Y,Xt,Yt,k_this):
    Y_new = completeLabel(X,Y,k=k_this, complete_which=0, self_weight=1)
    return HEPR(X,Y_new,Xt,Yt)

def HEPR_part(X,Y,Xt,Yt,k_this):
    Y_new = completeLabel(X,Y,k=k_this, complete_which=1, self_weight=1)
    return HEPR(X,Y_new,Xt,Yt)

def HEPR_noise(X,Y,Xt,Yt,k_this):
    # print(np.max(np.max(Y)),np.min(np.min(Y)))
    Y_new = completeLabel(X,Y,k=k_this, complete_which=2, self_weight=1)
    # print(np.max(np.max(Y_new)),np.min(np.min(Y_new)))
    return HEPR(X,Y_new,Xt,Yt)

if __name__ == '__main__':
    numdata = 1
    # datasnames = ["Yeast","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    datasnames = ["Yeast","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'HEPR_'
    kb = 20
    misrate = 0.3
    k_this = int(kb*misrate)
    for dataIdx in range(numdata):
        X,Y,Xt,Yt = rd.readData(dataIdx)
        for mode in ('mis','mis7','part','noise'):#
            print(mode)
            if(mode=='mis7'):
                Y2 = mistaking(np.array(Y),0.7,'mis')
            else:
                Y2 = mistaking(np.array(Y),misrate,mode)
            if(mode=='mis' or mode=='mis7'):
                prediction,traintime,testtime = HEPR_mis(X,Y2,Xt,Yt,k_this)
            if(mode=='part'):
                prediction,traintime,testtime = HEPR_part(X,Y2,Xt,Yt,k_this)
            if(mode=='noise'):
                prediction,traintime,testtime = HEPR_noise(X,Y2,Xt,Yt,k_this)
            saveResult(datasnames[dataIdx], algname+mode, evaluate(prediction, Yt), traintime, testtime)
