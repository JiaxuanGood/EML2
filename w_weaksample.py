from u_base import *
from w_weakcomplete import *

'''produce missing/partial/noise labels: 1 for positive, -1 for negative, 0 for missing'''
def mistaking(mat_, ratio, mode='mis'): # mode={'miss','part','noise'}
    x,y = np.shape(mat_)
    sum = x*y
    mat_ = (mat_-0.5)*2
    idx = random.sample(range(0,sum),int(sum*ratio))
    if(mode=='mis'):
        mat2 = rand_mis(mat_, idx)
    elif(mode=='part'):
        mat2 = rand_part(mat_, idx)
    elif(mode=='noise'):
        mat2 = rand_noise(mat_, idx)
    else:
        print('ERROR')
    mat2 = keep1(mat_,mat2)
    return mat2

def rand_mis(mat_,idx):
    y = np.shape(mat_)[1]
    mat2 = np.array(mat_)
    for i in range(len(idx)):
        a = int(idx[i]/y)
        b = idx[i] % y
        mat2[a][b] = 0
    return mat2
def rand_part(mat_,idx):
    y = np.shape(mat_)[1]
    mat2 = np.array(mat_)
    for i in range(len(idx)):
        a = int(idx[i]/y)
        b = idx[i] % y
        mat2[a][b] = 1
    return mat2
def rand_noise(mat_,idx):
    y = np.shape(mat_)[1]
    mat2 = np.array(mat_)
    for i in range(len(idx)):
        a = int(idx[i]/y)
        b = idx[i] % y
        mat2[a][b] = -1
    return mat2

'''for each label, if all positive labels are deleted, revise one of them'''
def keep1(mat_, mat2):
    for j in range(len(mat_[0])):
        if(np.sum(mat2[:,j]==1) == 0):
            idx = np.argwhere(mat_[:,j]==1).flatten()
            if(len(idx)==0):
                continue
            select_a = random.randint(0,len(idx)-1)
            mat2[idx[select_a]][j] = 1
        if(np.sum(mat2[:,j]==-1) == 0):
            idx = np.argwhere(mat_[:,j]==-1).flatten()
            if(len(idx)==0):
                continue
            select_a = random.randint(0,len(idx)-1)
            mat2[idx[select_a]][j] = -1
    return mat2

if __name__=="__main__":
    datasnames = ["Birds","CAL500","CHD_49","Enron","Flags","Foodtruck",
        "Genbase","GnegativeGO","GpositiveGO","Image","Langlog","Medical","PlantGO","Scene","Slashdot","Chemistry","Chess","Coffee","VirusGO","Yeast","Yelp"]
    rd = ReadData(datas=datasnames)
    for dataIdx in range(5,6):
        # print(dataIdx)
        X,Y,Xt,Yt = rd.readData(dataIdx)
        Y = mistaking(Y,0.3)
        saveMat(completeLabel(X,Y))
