from matplotlib import pyplot as plt
from preprocessing import preprocessing
from DSSC import DSSC
from kmeans import kmeans
from ncut import n_cut
from utils import flatten,brightness,gen_data
import numpy as np 
from PIL import Image
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--label',type = str, default = 'plane', help = 'The available label is horse, deer and plane')
    args = parser.parse_args()
    batch_size = 3
    images = preprocessing()

    true_labels = ["horse", "deer", "plane"]

    if args.label in true_labels:
        idx_of_labels = true_labels.index(args.label) # set the label of image
    else:
        raise Exception(f"The label {args.label} you enter is not available!")

    transpose_tensorimages = images.tensorimage()
    images = transpose_tensorimages[idx_of_labels].T
    centroids = 4
    iterations = 60
    
    fig, axs = plt.subplots(1,6, figsize=(15, 3), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.1)
    axs = axs.ravel()

    # kmeans ++ algorithm
    # kmeans++ with 5D clustering accuracy
    flatten_image_5d = flatten(gen_data(images,is_3d= False))
    # kmeans++ with 3D clustering accuracy
    flatten_image_3d = flatten(gen_data(images,is_3d= True))

    L_kmeans_5d, C_kmeans_5d = kmeans(flatten_image_5d,centroids, iterations,kmeanspp= True,normalise = True)
    L_kmeans_3d, C_kmeans_3d = kmeans(flatten_image_3d,centroids, iterations,kmeanspp= True,normalise = False)   

    L_kmeans_5d = L_kmeans_5d.reshape(images.shape)
    L_kmeans_3d = L_kmeans_3d.reshape(images.shape)

    axs[0].imshow(images), axs[0].set_title(f'Original {true_labels[idx_of_labels]} image'), axs[0].set_xticks([]), axs[0].set_yticks([])
    axs[1].imshow(L_kmeans_3d), axs[1].set_title(f'kmeans++ 3D of {true_labels[idx_of_labels]}'), axs[1].set_xticks([]), axs[1].set_yticks([])
    axs[2].imshow(L_kmeans_5d), axs[2].set_title(f'kmeans++ 5D of {true_labels[idx_of_labels]}'), axs[2].set_xticks([]), axs[2].set_yticks([])
    
    #ncut algorithm
    n_cut_image = n_cut(images,r=16,percent=0.098)
    keigenvectors= n_cut_image.eigenvs('sym') # methods: "rw" and "sym". 
                                            # rw is D^(-1)L
                                            # sym is D^(-1/2) L D^(-1/2)                  
    row = images.shape[0]
    l_ncut,C_ncut = kmeans(keigenvectors, centroids, iterations)
    
    n_cut_label = np.zeros((row*row, 3))
        
    for ind, Cs in enumerate(C_ncut):
        idxs = np.argwhere(np.all(l_ncut == Cs, axis=1)).flatten()
        n_cut_label[idxs] = np.mean(flatten_image_3d[idxs],axis=0) #an 1d array consists of index with Cs label

    n_cut_label = n_cut_label.reshape(images.shape)
    axs[3].imshow(n_cut_label), axs[3].set_title(f'Ncut of {true_labels[idx_of_labels]}'), axs[3].set_xticks([]), axs[3].set_yticks([])

    DSSC_img = DSSC(flatten_image_3d,p=20,Knn=19)
    keigenvectors= DSSC_img.eigenvs() 
    L_last,C_last = kmeans(keigenvectors, centroids, iterations)
    DSSC_label = np.zeros((row*row, 3))

    for ind, Cs in enumerate(C_last):
        idxs = np.argwhere(np.all(L_last == Cs, axis=1)).flatten()
        DSSC_label[idxs] = np.mean(flatten_image_3d[idxs],axis=0) #an 1d array consists of index with Cs label

    DSSC_label = DSSC_label.reshape(images.shape)
    axs[4].imshow(DSSC_label), axs[4].set_title(f'DSSC of {true_labels[idx_of_labels]}'), axs[4].set_xticks([]), axs[4].set_yticks([])

    my_pic = Image.open(f"test{idx_of_labels}.png")
    my_pic = np.array(my_pic,dtype = np.uint8)[:,:,:3]
    axs[5].imshow(my_pic), axs[5].set_title(f'Our result of {true_labels[idx_of_labels]}'), axs[5].set_xticks([]), axs[5].set_yticks([])

    # plt.savefig(f"show_{idx_of_labels+1}.pdf", bbox_inches = 'tight',pad_inches = 0)
    plt.show()