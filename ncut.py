from preprocessing import preprocessing
from kmeans import kmeans
import matplotlib.pyplot as plt
import numpy as np 
from utils import brightness,flatten

class n_cut:
    def __init__(self,image,r,percent):
        # setting the parameters
        self.r = r
        self.percent = percent
        self.row = image.shape[0]
        self.col = image.shape[1]
        self.image = image
        self.K = 4
        self.flatten = flatten(image)
        self.W, self.D = self.WD()

    def WD(self):
        """
        generate the W weight matrix and D degree matrix
        output: D and W
        """
        row_ = self.flatten.shape[0]
        F = np.zeros((row_,row_)) # F n*n
        X = np.zeros((row_,row_)) # X n*n
        for i in range(row_):
            for j in range(row_):
                if i != j:
                    coor_i = i//self.row, i%self.row
                    coor_j = j//self.row, j%self.row      # map the index (i,j) to (x,y) coordinate on the image
                    if np.linalg.norm( np.array(coor_i) - np.array(coor_j)) < self.r:
                        X[i,j] = np.linalg.norm( np.array(coor_i) - np.array(coor_j))
                        F[i,j] = np.linalg.norm(brightness(self.image[coor_i]) - brightness(self.image[coor_j]))
                    
        sigma_i = self.percent * (np.max(F) - np.min(F)) # calculate sigma_i = precentage * d(F)
        sigma_x = self.percent * (np.max(X) - np.min(X)) # calculate sigma_x = precentage * X(F)

        W = np.exp(-np.power(F,2)/sigma_i**2,where=F!=0) * np.exp(-np.power(X,2)/sigma_x**2,where=X!=0)
        D = np.diag(np.sum(W,axis=1)) # fill the values on the diagonal matrix.
        return W, D

    def laplacian(self,method):
        if method == 'sym':
            L = np.diag(np.power(np.diag(self.D),-0.5)) @ (self.D-self.W) @ np.diag(np.power(np.diag(self.D),-0.5))
        elif method == 'rw':
            L = np.diag(np.power(np.diag(self.D),-1)) @ (self.D-self.W)
        return L
    
    def eigenvs(self, method):
        """
        input: whether the eigen vectors need to be normalised
        output: the eigen column vector 
        """
        L = self.laplacian(method)
        _, v = np.linalg.eigh(L)
        v = v[:,:self.K]
        if method == 'sym':
            v /= np.linalg.norm(v) #scale to unit length.
        return v
    
if __name__ == '__main__':
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
        n_cut_image = n_cut(images,r=16,percent=0.098)
        keigenvectors= n_cut_image.eigenvs('sym') # methods: "rw" and "sym". 
                                                # rw is D^(-1)L
                                                # sym is D^(-1/2) L D^(-1/2)                  
        row = images.shape[0]
        flatten_image = flatten(images)
        L_last,C_last = kmeans(keigenvectors, centroids, iterations)
        color_label = np.zeros((row*row, 3))
        
        for ind, Cs in enumerate(C_last):
            idxs = np.argwhere(np.all(L_last == Cs, axis=1)).flatten()
            color_label[idxs] = np.mean(flatten_image[idxs],axis=0) #an 1d array consists of index with Cs label

        color_label = color_label.reshape(images.shape)
        axs[i].imshow(images), axs[i].set_title('The original image')
        axs[i+batch_size].imshow(color_label), axs[i+batch_size].set_title('Corresponding ncut image')
    plt.show()