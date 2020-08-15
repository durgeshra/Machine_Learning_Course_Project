import numpy as np
from numpy import random as rand
import utils 
from scipy import sparse as sps
import os
utils.__file__
# DO NOT CHANGE THE NAME OF THIS METHOD OR ITS INPUT OUTPUT BEHAVIOR

# INPUT CONVENTION
# X: n x d matrix in csr_matrix format containing d-dim (sparse) features for n test data points
# k: the number of recommendations to return for each test data point in ranked order

# OUTPUT CONVENTION
# The method must return an n x k numpy nd-array (not numpy matrix or scipy matrix) of labels with the i-th row 
# containing k labels which it thinks are most appropriate for the i-th test point. Labels must be returned in 
# ranked order i.e. the label yPred[i][0] must be considered most appropriate followed by yPred[i][1] and so on

# CAUTION: Make sure that you return (yPred below) an n x k numpy (nd) array and not a numpy/scipy/sparse matrix
# The returned matrix will always be a dense matrix and it terribly slows things down to store it in csr format
# The evaluation code may misbehave and give unexpected results if an nd-array is not returned



def using_tocoo(x):
    f1 = open("PfastreXML/Data/test_data.X","w+")
    (n, d) = x.shape
    s=""
    s+=("%d %d"%(n,d))
    cx = x.tocoo()
    prev = -1
    for (i,j,v) in zip(cx.row,cx.col,cx.data):
        if(i == prev):
            s+=(" %d:%f"%(j,v))
        else:
            while(prev!=i):
                s+=("\n")
                if(prev%2000==0):
                    f1.write(s)
                    s=""
                prev+=1
            s+=("%d:%f"%(j,v))
    s+=("\n")
    f1.write(s)
    f1.close()
    return


def getReco( X, k ):
    # Find out how many data points we have
    (n, d) = X.shape
    using_tocoo(X)
    os.system("./predict_script.sh")
    fi=open('PfastreXML/Results/score_mat.txt','r')    
    n,d=tuple(fi.readline().split(' '))
    n,d =int(n),int(d)
    arr = np.ndarray((n,k),dtype=int)
    for j,i in enumerate(fi.readlines()):
        l=list(map(lambda x:tuple(x.split(':')),i.split(' ')))
        arr[j]=list(map(lambda x:int(x[0]),(sorted(l,key=lambda x:float(x[1]),reverse=True))))[:k]
    fi.close()
    return arr
