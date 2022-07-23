from preprocessing import preprocessing
from DSSC import DSSC
from kmeans import kmeans
from ncut import n_cut
from scipy.optimize import linear_sum_assignment as hungarian
import numpy as np 
from PIL import Image
from utils import flatten,brightness,gen_data,pixeltolabel
from sklearn.metrics.cluster import normalized_mutual_info_score as NMI
from sklearn.metrics import fowlkes_mallows_score as fscore

def hungarian_matching(y_true,y_pred):
    """
    input: y_true, y_pred
    output: w,ind
    """
    D = max(y_pred.max(), y_true.max()) + 1
    w = np.zeros((D, D))
    for i in range(y_pred.size):
        w[y_pred[i], y_true[i]] += 1
    cost_matrix = w.max() - w
    ind = hungarian(cost_matrix)
    ind = np.array(list(zip(*ind))) # calculate the relation between two labels
    return w,ind

def acc(y_true, y_pred):
    """
    Calculate clustering accuracy.
    # Arguments
        y_true: labels,
        y_pred: predicted labels,
    # Return
        accuracy, in [0,1]
    """
    w,ind = hungarian_matching(y_true,y_pred)
    return sum([w[i, j] for i, j in ind]) * 1.0 / y_pred.size

def NMI_score(y_true, y_pred):
    """
    Calculate the NMI score of clustering
    # Arguments:
        y_true: labels,
        y_pred: predicted labels
    """
    w,ind = hungarian_matching(y_true,y_pred)
    y_pred_ = np.zeros(y_pred.shape) # calculate the actual y_pred
                                    # and set as y_pred_
    for i, j in ind:
        y_pred_[np.argwhere(y_pred == j)] = i
    return NMI(y_true,y_pred_)

def f_score(y_true, y_pred):
    """
    Calculate the V-measure of clustering
    # Arguments:
        y_true: labels,
        y_pred: predicted labels
    """
    w,ind = hungarian_matching(y_true,y_pred)
    y_pred_ = np.zeros(y_pred.shape) # calculate the actual y_pred
                                    # and set as y_pred_
    for i, j in ind:
        y_pred_[np.argwhere(y_pred == j)] = i
    return fscore(y_true,y_pred_)

if __name__ == '__main__':
    batch_size = 3
    images = preprocessing()
    transpose_tensorimages = images.tensorimage()
    centroids = 4
    iterations = 60

    accuracy_kmean3d = 0
    accuracy_kmean5d = 0
    accuracy_ncut = 0
    accuracy_DSSC = 0

    iou_kmeans3d = 0
    iou_kmeans5d = 0
    iou_ncut = 0
    iou_DSSC = 0

    NMI_kmeans3d = 0
    NMI_kmeans5d = 0
    NMI_ncut = 0
    NMI_DSSC  =0 

    f_score_kmeans3d = 0
    f_score_kmeans5d = 0
    f_score_ncut = 0
    f_score_DSSC = 0

    for i in range(batch_size):
        images = transpose_tensorimages[i].T

        # kmeans ++ algorithm
        # kmeans++ with 5D clustering accuracy
        flatten_image_5d = flatten(gen_data(images,is_3d= False))
        # kmeans++ with 3D clustering accuracy
        flatten_image_3d = flatten(gen_data(images,is_3d= True))
        img = Image.open(f"test{i}.png")
        img = np.asarray(img)[:,:,:3]

        y_true = pixeltolabel(flatten(img)) # y_true is manually labeled

        L_kmeans_5d, C_kmeans_5d = kmeans(flatten_image_5d,centroids, iterations,kmeanspp= True,normalise = True)
        L_kmeans_3d, C_kmeans_3d = kmeans(flatten_image_3d,centroids, iterations,kmeanspp= True,normalise = False)   
        label_kmeans_3d = pixeltolabel(L_kmeans_3d)
        label_kmeans_5d = pixeltolabel(L_kmeans_5d)
        accuracy_kmean3d += acc(y_true,label_kmeans_3d)
        accuracy_kmean5d += acc(y_true,label_kmeans_5d)
        NMI_kmeans3d += NMI_score(y_true,label_kmeans_3d)
        NMI_kmeans5d += NMI_score(y_true,label_kmeans_5d)
        f_score_kmeans3d += f_score(y_true,label_kmeans_3d)
        f_score_kmeans5d += f_score(y_true,label_kmeans_5d)

        #ncut algorithm
        n_cut_image = n_cut(images,r=16,percent=0.098)
        keigenvectors= n_cut_image.eigenvs('sym') # methods: "rw" and "sym". 
                                                # rw is D^(-1)L
                                                # sym is D^(-1/2) L D^(-1/2)                  
        row = images.shape[0]
        l_ncut,C_ncut = kmeans(keigenvectors, centroids, iterations)
        label_ncut = pixeltolabel(l_ncut)
        accuracy_ncut += acc(y_true,label_ncut)
        NMI_ncut += NMI_score(y_true,label_ncut)
        f_score_ncut += f_score(y_true,label_ncut)

        # DSSC
        row = images.shape[0]
        DSSC_img = DSSC(flatten_image_3d,p=20,Knn=19)
        keigenvectors= DSSC_img.eigenvs() 
        l_DSSC, c_DSSC = kmeans(keigenvectors, centroids, iterations)
        label_DSSC = pixeltolabel(l_DSSC)
        accuracy_DSSC += acc(y_true,label_DSSC)
        NMI_DSSC +=  NMI_score(y_true,label_DSSC)
        f_score_DSSC += f_score(y_true,label_DSSC)

    # kmeans++ with 3D clustering accuracy
    print(f"Accuracy of Kmeans 3d:{accuracy_kmean3d/batch_size}")
    # kmeans++ with 5D clustering accuracy
    print(f"Accuracy of Kmeans 5d:{accuracy_kmean5d/batch_size}")
    # Ncut clustering algorithm accuracy
    print(f"Accuracy of normalised cut:{accuracy_ncut/batch_size}")
    # DSSC clustering algorithm accuracy
    print(f"Accuracy of DSSC clustering algorithm:{accuracy_DSSC/batch_size}")

    print()

    # kmeans++ with 3D clustering IOU
    print(f"f_score of Kmeans 3d:{f_score_kmeans3d/batch_size}")
    # kmeans++ with 5D clustering IOU
    print(f"f_score of Kmeans 5d:{f_score_kmeans5d/batch_size}")
    # Ncut clustering algorithm IOU
    print(f"f_score of normalised cut:{f_score_ncut/batch_size}")
    # DSSC clustering algorithm IOU
    print(f"f_score of DSSC clustering algorithm:{f_score_DSSC/batch_size}")

    print()

    # kmeans++ with 3D clustering NMI
    print(f"NMI of Kmeans 3d:{NMI_kmeans3d/batch_size}")
    # kmeans++ with 5D clustering NMI
    print(f"NMI of Kmeans 5d:{NMI_kmeans5d/batch_size}")
    # Ncut clustering algorithm NMI
    print(f"NMI of normalised cut:{NMI_ncut/batch_size}")
    # DSSC clustering algorithm NMI
    print(f"NMI of DSSC clustering algorithm:{NMI_DSSC/batch_size}")

        



        