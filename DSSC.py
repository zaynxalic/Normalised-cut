from preprocessing import preprocessing
import matplotlib.pyplot as plt
import numpy as np
from kmeans import kmeans
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra
from utils import flatten,brightness


class DSSC:
    def __init__(self,flatten_image,p,Knn):
        self.p = p
        self.flatten_image = flatten_image
        self.length = self.flatten_image.shape[0]
        self.A = kneighbors_graph(self.flatten_image,Knn, mode='connectivity', include_self=True).toarray()
        self.Den = self.Density_matrix()
        self.W = 1.0/(self.Den + 1.0)
        self.D = np.diag(np.sum(self.W,axis=1))
        self.L = self.laplacian()
        self.K = 4

    def Density_matrix(self):
        """
        Calculate the matrix L(xi,xj) = (e^{p * dist(xi,xj)} - 1)^{1/p}
        """    
        L = np.zeros(self.A.shape)
        for i in range(self.length):
            for j in range(self.length):
                if self.A[i,j] == 1.0:
                    coor_i = brightness(self.flatten_image[i])
                    coor_j = brightness(self.flatten_image[j])
                    L[i,j] = np.power(
                    np.exp(self.p* np.linalg.norm(coor_i - coor_j))-1.0, 
                    1.0/self.p)

        dist_matrix = dijkstra(csgraph=csr_matrix(L), directed=False, return_predecessors=False)
        dist_matrix[dist_matrix == np.inf] = 1e-12 # set infinity value to 1e-12.
        return dist_matrix

    def laplacian(self):
        L = np.eye(self.length) - np.diag(np.power(np.diag(self.D),-0.5)) @ self.W @ np.diag(np.power(np.diag(self.D),-0.5))
        return L
        
    def eigenvs(self):
        _, v = np.linalg.eigh(self.L)
        v = v[:,:self.K]
        return v
        
if __name__ == '__main__':
    batch_size = 3
    images = preprocessing()
    transpose_tensorimages = images.tensorimage()
    centroids = 4
    iterations = 40
    
    fig, axs = plt.subplots(2,batch_size, figsize=(15, 6), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()

    for i in range(batch_size):
        images = transpose_tensorimages[i].T
        flatten_images = flatten(images)
        row = images.shape[0]
        DSSC_img = DSSC(flatten_images,p=20,Knn=19)
        keigenvectors= DSSC_img.eigenvs() 
        L_last,C_last = kmeans(keigenvectors, centroids, iterations)
        color_label = np.zeros((row*row, 3))

        for ind, Cs in enumerate(C_last):
            idxs = np.argwhere(np.all(L_last == Cs, axis=1)).flatten()
            color_label[idxs] = np.mean(flatten_images[idxs],axis=0) #an 1d array consists of index with Cs label

        color_label = color_label.reshape(images.shape)
        axs[i].imshow(images), axs[i].set_title('The original image')
        axs[i+batch_size].imshow(color_label), axs[i+batch_size].set_title('Corresponding DSSC image')
    plt.show()