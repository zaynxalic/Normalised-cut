import numpy as np 

def gen_data(X, is_3d = True):
    """
    generate the data 3d data if the data is in 3d.
    geneate the data with coordinate data if the data is in 5d.
    """
    if is_3d:
        return X 
    X_ = np.zeros((X.shape[0],X.shape[1],5))
    for i in range(X.shape[0]):
        for j in range(X.shape[1]):
            X_[i,j] = np.r_[X[i,j],i,j]
    return X_

def brightness(RGB):
    """
    input: a 1D RGB numpy array [R,G,B]
    output: the brightness of the image
    """
    # l1 = (RGB[0]+RGB[1]+RGB[2])/3
    # l2 = RGB[0] - RGB[2]/2
    # l3 = (2*RGB[0] - RGB[1] - RGB[2])/4
    # return np.asarray([l1,l2,l3])
    return np.mean(RGB)

def flatten(X):
    """
    flatten the image to (N,3)
    """
    return X.reshape(-1,X.shape[-1])

def pixeltolabel(X):
    """
    input: pixel of each clusters
    Change clusters to different labels
    """
    u,indices = np.unique(X,axis=0,return_inverse=True)
    return indices