#imports
# import readFaceImages
import numpy as np
# import matplotlib.pyplot as plt
# import math
# from tabulate import tabulate
from sklearn.cluster import KMeans
# from sklearn.neighbors import KDTree, KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classifier import fake_dists

def func(descriptors):
    num_clusters = len(descriptors, X_train)

    descriptors = np.zeros((5, 20))

    # fit kmeans model to subset of training data
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=300).fit(X_train)
    clusters = kmeans.cluster_centers_





    # Calculate a histogram for each image in the training set. The K means algorithm
    # which now knows the centroids, predicts which centroid each SIFT descriptor
    # belongs to aka perform NN on the centroids.


    # iterate over all train images and create a histogram for each
    # image by using NN to assign each SIFT descriptor to a centroid
    histograms = []
    for i in range(len(X_train)):
        im = train_D[i]
        tree = KDTree(clusters)
        distance, indices = tree.query(im, k=1)   
        histogram = np.histogram(indices, bins=np.arange(1, 122))
        histograms.append(histogram[0])
        
    # Each training image has a histogram representing roughly how many visual words that image has
    # of whatever type of visual words there are in the total set of images.





    # Fit K neighbors classifier with the histograms and the labels of the training data.
    neighbors = KNeighborsClassifier().fit(histograms, train_gs[0])





    # Load in testing data, build histograms for the testing data and then use k means classifier to
    # predict on the testing histograms. This will yield label predictions for every test image.

    # iterate over all test images and create a histogram for each image
    X_test = test_D
    histograms_test = []
    for i in range(len(X_test)):
        image = X_test[i]
        
        # use NN to assign each SIFT descriptor to a centroid (build histogram)
        distance, indices = tree.query(image, k=1)     
        histogram = np.histogram(indices, bins=np.arange(1, 122))
        histogram = histogram[0].reshape(1,-1)
        histograms_test.append(histogram)




    labels = []
    for i in range(len(histograms_test)):
        labels.append(neighbors.predict(histograms_test[i]))





    # matrix = confusion_matrix(test_gs[0], labels)
    # display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=neighbors.classes_)
    # display.plot()
    # plt.show()