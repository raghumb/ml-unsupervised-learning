import numpy
import colorsys
import random
import os
import numpy as np
from matplotlib.mlab import PCA as mlabPCA
from matplotlib import pyplot as plt

# Referrred fr0m https://stackoverflow.com/questions/26645642/plot-multi-dimension-cluster-to-2d-plot-python
def get_colors(num_colors):
    """
    Function to generate a list of randomly generated colors
    The function first generates 256 different colors and then
    we randomly select the number of colors required from it
    num_colors        -> Number of colors to generate
    colors            -> Consists of 256 different colors
    random_colors     -> Randomly returns required(num_color) colors
    """
    colors = []
    random_colors = []
    # Generate 256 different colors and choose num_clors randomly
    for i in numpy.arange(0., 360., 360. / 256.):
        hue = i / 360.
        lightness = (50 + numpy.random.rand() * 10) / 100.
        saturation = (90 + numpy.random.rand() * 10) / 100.
        colors.append(colorsys.hls_to_rgb(hue, lightness, saturation))

    for i in range(0, num_colors):
        random_colors.append(colors[random.randint(0, len(colors) - 1)])
    return random_colors

# Referrred fr0m https://stackoverflow.com/questions/26645642/plot-multi-dimension-cluster-to-2d-plot-python
def random_centroid_selector(total_clusters , clusters_plotted):
    """
    Function to generate a list of randomly selected
    centroids to plot on the output png
    total_clusters        -> Total number of clusters
    clusters_plotted      -> Number of clusters to plot
    random_list           -> Contains the index of clusters
                             to be plotted
    """
    random_list = []
    for i in range(0 , clusters_plotted):
        random_list.append(random.randint(0, total_clusters - 1))
    return random_list

# Referrred fr0m https://stackoverflow.com/questions/26645642/plot-multi-dimension-cluster-to-2d-plot-python
def plot_cluster(kmeansdata, centroid_list, label_list , num_cluster, title, prefix):
    """
    Function to convert the n-dimensional cluster to 
    2-dimensional cluster and plotting 50 random clusters
    file%d.png    -> file where the output is stored indexed
                     by first available file index
                     e.g. file1.png , file2.png ...
    """
    mlab_pca = mlabPCA(kmeansdata)
    cutoff = mlab_pca.fracs[1]
    users_2d = mlab_pca.project(kmeansdata, minfrac=cutoff)
    centroids_2d = mlab_pca.project(centroid_list, minfrac=cutoff)


    colors = get_colors(num_cluster)
    plt.title(title)
    plt.figure()
    plt.xlim([users_2d[:, 0].min() - 3, users_2d[:, 0].max() + 3])
    plt.ylim([users_2d[:, 1].min() - 3, users_2d[:, 1].max() + 3])

    # Plotting 50 clusters only for now
    random_list = random_centroid_selector(num_cluster , 50)

    # Plotting only the centroids which were randomly_selected
    # Centroids are represented as a large 'o' marker
    for i, position in enumerate(centroids_2d):
        if i in random_list:
            plt.scatter(centroids_2d[i, 0], centroids_2d[i, 1], marker='o', c=colors[i], s=100)


    # Plotting only the points whose centers were plotted
    # Points are represented as a small '+' marker
    for i, position in enumerate(label_list):
        if position in random_list:
            plt.scatter(users_2d[i, 0], users_2d[i, 1] , marker='+' , c=colors[position])

    filename = 'images/' + prefix
    i = 0
    while True:
        if os.path.isfile(filename + str(i) + ".png") == False:
            #new index found write file and return
            plt.savefig(filename + str(i) + ".png")
            break
        else:
            #Changing index to next number
            i = i + 1
    return

# Referrred fr0m https://stackoverflow.com/questions/26645642/plot-multi-dimension-cluster-to-2d-plot-python
def plot_cluster_wo_centroid(kmeansdata, label_list , num_cluster, title, prefix):
    """
    Function to convert the n-dimensional cluster to 
    2-dimensional cluster and plotting 50 random clusters
    file%d.png    -> file where the output is stored indexed
                     by first available file index
                     e.g. file1.png , file2.png ...
    """
    mlab_pca = mlabPCA(kmeansdata)
    cutoff = mlab_pca.fracs[1]
    users_2d = mlab_pca.project(kmeansdata, minfrac=cutoff)
    #centroids_2d = mlab_pca.project(centroid_list, minfrac=cutoff)


    colors = get_colors(num_cluster)
    plt.title(title)
    plt.figure()
    plt.xlim([users_2d[:, 0].min() - 3, users_2d[:, 0].max() + 3])
    plt.ylim([users_2d[:, 1].min() - 3, users_2d[:, 1].max() + 3])

    # Plotting 50 clusters only for now
    random_list = random_centroid_selector(num_cluster , 50)

    

    # Plotting only the points whose centers were plotted
    # Points are represented as a small '+' marker
    for i, position in enumerate(label_list):
        if position in random_list:
            plt.scatter(users_2d[i, 0], users_2d[i, 1] , marker='+' , c=colors[position])

    filename = 'images/' + prefix
    i = 0
    while True:
        if os.path.isfile(filename + str(i) + ".png") == False:
            #new index found write file and return
            plt.savefig(filename + str(i) + ".png")
            break
        else:
            #Changing index to next number
            i = i + 1
    return    


# Referred from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def draw_vector(v0, v1, ax=None):
    ax = ax or plt.gca()
    arrowprops=dict(arrowstyle='->',
                    linewidth=2,
                    shrinkA=0, shrinkB=0)
    ax.annotate('', v1, v0, arrowprops=arrowprops)

# Referred from https://jakevdp.github.io/PythonDataScienceHandbook/05.09-principal-component-analysis.html
def plot_pca_vector(X , pca):
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2)
    for length, vector in zip(pca.explained_variance_, pca.components_):
        v = vector * 3 * np.sqrt(length)
        draw_vector(pca.mean_, pca.mean_ + v)
    plt.axis('equal')
    plt.title('PCA Variance')
    filename = 'images/' + 'PCA-Variance'
    plt.savefig(filename + ".png")


