

__author__ = 'haitham'
import numpy as np
def get4D( X):
        sh  = np.asarray(X.shape)
        sh.resize((1,4))
        sh = sh[0]
        sh[sh==0]=1
        X = X.reshape(sh[0], sh[1], sh[2], sh[3])
        X= np.asarray(X,dtype=np.float32)
        return X

