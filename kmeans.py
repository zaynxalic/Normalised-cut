from preprocessing import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from utils import flatten,gen_data
import argparse

# referenced from comp6670 assignment 2.
def initialiseparameters(m, X, kmeanspp=True):
    """
    initialize the centriods where:
    m = #centriods
    X = the color channels of the flattened image
    kmeanspp = check whether kmeans
    Return:
    the intialised kmeans++ clusters C
    """
    np.random.seed(1) #set random seed to 1
    if kmeanspp:
        # calculate the flattened length of images which is 32 * 32
        size = X.shape[0]
        # calculate the dimension of Kmeans
        dim = X.shape[1]
        # the size of cluster is #clusters * length
        C = np.zeros((m,dim))
        X_ = X.copy()

        #randomly choose a centriods
        index = np.random.randint(0,size-1)
        #add the first centeriod
        first_centroid = X_[index]
        C[0] = first_centroid
        # set a list whose element is its index.
        choseindex = np.arange(size)

        # generate the minimum_distance
        for c_num in range(1,m):
            dist = np.zeros((size,c_num))
            for idx in range(c_num):
                dist[:,idx] = np.power(np.linalg.norm(X_ - C[idx],ord=2, axis=1),2)
            
            # calculate the min_dist D^2
            min_dist = np.min(dist,axis=1)

            # let D^4/ sumof(D^4) as the possibility, 
            min_dist = min_dist**4/np.sum(min_dist**4)
            #chosing the value based on the probablity
            chosen_value = np.random.choice(choseindex,1,p=min_dist)[0]
            # chosing the value if the value is the maximum
            maxValue = X_[chosen_value]
            C[c_num] = maxValue
    else:
        """
        The original of Kmeans algorithm which random select 4 clusters as centers
        """
        size = X.shape[0]
        choseindex = np.arange(size)
        idxs = np.random.choice(choseindex,m)
        C = X[idxs]
    return C

def E_step(C, X):
    L = np.zeros(X.shape)
    for i in range(X.shape[0]):
        minidist = np.linalg.norm(X[i]-C, ord =2, axis=1)
        minindex = np.argmin(minidist)
        L[i] = C[minindex]
    return L

def M_step(C, X, L):
    """
    calculate the center of clusters
    """
    C_copy = np.zeros(C.shape)
    for i in range(C.shape[0]):
        idxs = np.argwhere(np.all(L == C[i], axis=1)).flatten()
        C_copy[i] = np.mean(X[idxs],axis = 0)
    return C_copy

def kmeans(X, m, iter,kmeanspp=True, normalise = False):
    """
    X means the data inputs,
    m means the number of centroids
    iter equals to the number of iterations
    """
    min = None
    max = None 

    # normalise the X data
    if normalise:
        min = np.min(X,axis=0)
        max = np.max(X,axis=0)
        X = (X - min)/(max-min)

    L = np.zeros(X.shape)
    C = np.zeros((m, X.shape[1]))
    # initialise the clusters using kmeans++ or kmeans algorithms
    C = initialiseparameters(m, X,kmeanspp)
    for _ in range(iter):
        L = E_step(C,X)
        C = M_step(C,X,L)

    # denormalise the data
    if normalise:
        L = L*(max-min) + min
        C = C*(max-min) + min
        L = L[:,:3]
        C = C[:,:3]
    return L,C
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dim', type = int, default = 3, help = 'dimension should be 3 or 5, default is 3')
    args = parser.parse_args()
    _dim = args.dim
    batch_size = 3
    images = preprocessing()
    transpose_tensorimages = images.tensorimage()
    centroids = 4
    iterations = 60
    fig, axs = plt.subplots(2,batch_size, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    # show the images in batch_size * 2
    for i in range(batch_size):
        images = transpose_tensorimages[i].T
        flatten_image = flatten( gen_data(images,is_3d= (_dim == 3)))
        L_last, C_Last = kmeans(flatten_image,centroids, iterations,kmeanspp= True,normalise = True)        
        L_last = L_last.reshape(images.shape)
        axs[i].imshow(images), axs[i].set_title('The original image')
        axs[i+batch_size].imshow(L_last),axs[i+batch_size].set_title(f'Corresponding kmeans++ {_dim}d image')

    plt.show()