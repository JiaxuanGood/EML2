from sklearn import model_selection
from operator import itemgetter
from u_base import *
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
from skmultilearn.problem_transform import BinaryRelevance

def br_predict(X,Y,Xt):
    br = BinaryRelevance(classifier=base_cls())
    br.fit(X,Y)
    prediction = br.predict(Xt).todense()
    return np.array(prediction)

def fuzzyDist(X, Xt_a, beta=5):
    fDists = pairwise_distances([Xt_a], X)
    fDists = np.array(fDists)
    return np.exp(-beta*(fDists*fDists))

def calF1(V_and_D_memdeg,V_rem_D_memdeg,D_rem_V_memdeg,N_memdeg):
    tp_memdeg,fn_memdeg,fp_memdeg = [],[],[]
    for i in range(len(N_memdeg)):
        tp_memdeg.append(min(V_and_D_memdeg[i],N_memdeg[i]))
        fn_memdeg.append(min(V_rem_D_memdeg[i],N_memdeg[i]))
        fp_memdeg.append(min(D_rem_V_memdeg[i],N_memdeg[i]))
    tp = np.sum(tp_memdeg)  # cardinality of a fuzzy set
    fn = np.sum(fn_memdeg)
    fp = np.sum(fp_memdeg)
    if(tp+fp+fn==0):
        return 0.0
    else:
        return 2*tp/(2*tp+fp+fn)

def CCkNN(X_org,Y,Xt, all_labels_order, k_CCkNN=5):
    cap_test = np.shape(Xt)[0]
    num_label = np.shape(Y)[1]
    results = np.zeros((cap_test,num_label))

    distanceses_eu = pairwise_distances(Xt,X_org)    # (num_train,num_feature),(1,num_feature) -> (num_train,1)
    distanceses = [[distanceses_eu[i][j]**2 for j in range(len(distanceses_eu[i]))] for i in range(len(distanceses_eu))]
    distanceses = np.array(distanceses)
    for i in range(cap_test):
        Xt_a = Xt[i]
        X = X_org
        distances = distanceses[i].flatten()
        labels = all_labels_order[i]
        predict_tmp_for_this_label = 0
        for q in range(num_label):
            distances = update_distances(distances,Y,predict_tmp_for_this_label, labels[q])
            indexs = np.argsort(distances)[:k_CCkNN]
            prediction = np.sum(Y[indexs,labels[q+1]])/k_CCkNN
            results[i][labels[q+1]] = prediction
            predict_tmp_for_this_label = prediction
    return results

def update_distances(distances,Y,Y_this_label, this_label):
    if(this_label == -1):
        return distances
    tmp = Y[:,this_label]-Y_this_label
    distance_add = [num*num for num in tmp]
    distances += np.array(distance_add)
    return distances

# K_num_baseLearners = 10 #defination
# k_CCkNN = 10 #{1,3,5,7,9,11}
# beta_order = 5 #{1,2,3,4,5,6,7,8,9,10}
# k_labelorder_num_neibors = 10
class DECC():
    def __init__(self,num_base=10,num_kcc=10,num_neibor=10,beta_order=5):
        self.num_base = num_base
        self.num_kcc = num_kcc
        self.num_neibor = num_neibor
        self.beta_order = beta_order
    def train(self, traX, traY):
        self.num_label = np.shape(traY)[1]
        self.X_org,self.Y_org = traX,traY
        self.obj_nbs = []
        self.Xv_s,self.Yv_s,self.YvBR_s = [],[],[]
        for i in range(self.num_base):
            X,Xv,Y,Yv = model_selection.train_test_split(traX,traY, train_size=0.66)  # train->(train1,valid)
            # V_org = np.hstack((Xv,Yv))
            YvBR = br_predict(X,fill1(Y),Xv)
            findNb = NearestNeighbors(n_neighbors=self.num_neibor, algorithm='ball_tree')
            findNb.fit(Xv)
            self.obj_nbs.append(findNb)
            self.Xv_s.append(Xv)
            self.Yv_s.append(Yv)
            self.YvBR_s.append(YvBR)
    def test(self,Xt):
        cap_test = len(Xt)
        predictions = np.zeros((cap_test,self.num_label))
        for i in range(self.num_base):
            findNb = self.obj_nbs[i]
            Xv,Yv,YvBR = self.Xv_s[i],self.Yv_s[i],self.YvBR_s[i]
            distances,indices = findNb.kneighbors(Xt)
            all_labels_order = []
            for i in range(cap_test):
                # # Vq,Dq,Nq have same members but different member_degrees, so only member_degrees calculation is enough
                Xv_neibor_this = Xv[indices[i]]
                Yv_neibor_this = Yv[indices[i]]
                # N_V_D_mem = np.hstack((Xv_neibor_this, Yv_neibor_this))
                YvBR_neibor_this = YvBR[indices[i]]
                N_memdeg = fuzzyDist(Xv_neibor_this, Xt[i], self.beta_order)[0]
                F1 = []
                for q in range(self.num_label):
                    V_and_D_memdeg = np.zeros(self.num_neibor)
                    V_rem_D_memdeg = np.zeros(self.num_neibor)
                    D_rem_V_memdeg = np.zeros(self.num_neibor)
                    
                    for ii in range(self.num_neibor):
                        if(Yv_neibor_this[ii][q]==1 and YvBR_neibor_this[ii][q]==1):
                            V_and_D_memdeg[ii] = 1
                        else:
                            if(Yv_neibor_this[ii][q]==1):
                                V_rem_D_memdeg[ii] = 1
                            if(YvBR_neibor_this[ii][q]==1):
                                D_rem_V_memdeg[ii] = 1
                    F1.append(calF1(V_and_D_memdeg,V_rem_D_memdeg,D_rem_V_memdeg,N_memdeg))
                F1_indices, F1_sorted = zip(*sorted(enumerate(-np.array(F1)), key=itemgetter(1)))
                all_labels_order.append(F1_indices)
            all_labels_order = np.array(all_labels_order)
            tmp = np.zeros((cap_test,1))-1
            all_labels_order = np.hstack((tmp,all_labels_order))

            predictions_this = CCkNN(self.X_org,self.Y_org,Xt, all_labels_order.astype(int))
            predictions += np.array(predictions_this)
        output = predictions/self.num_base
        return output

if __name__ == '__main__':
    numdata = 1
    datasnames = ["Yeast","Birds","CAL500","CHD_49","Enron","Flags","Foodtruck","GnegativeGO","GpositiveGO","Image","Langlog"]
    rd = ReadData(datas=datasnames,genpath='E:/multiLabel/DATA/arff/')
    # rd = ReadData(datas=datasnames,genpath='data/')
    algname = 'DECC'

    '''train-test'''
    for dataIdx in range(numdata):
        print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        print(np.shape(X),np.shape(Y),np.shape(Xt),np.shape(Yt))
        start_time = time()
        learner = DECC()
        learner.train(X,Y)
        mid_time = time()
        prediction = learner.test(Xt)
        saveResult(datasnames[dataIdx], algname, evaluate(prediction, Yt), (mid_time-start_time), (time()-mid_time))