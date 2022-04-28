#imports
# import readFaceImages
import numpy as np
# import matplotlib.pyplot as plt
# import math
# from tabulate import tabulate
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree, KNeighborsClassifier
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from classifier import create_descriptor, fake_dists


"""
params:
X_train = N 20-d descriptors for ground truth gestures, values in range [0,1] from landmark detector
X_test = M 20-d descriptors for test images, values in range [0,1] from landmark detector

output:
pred_labels = M-d array containing predicted labels of test images

This takes in train and test data (20-d descriptors), creates N clusters, 
and returns the predicted labels of each test image
"""
def predict_labels(X_train, X_test):
    num_clusters = len(X_train)
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=300).fit(X_train)
    pred_labels = kmeans.predict(X_test)

    # print(kmeans.labels_)
    # print(pred_labels)
    # print(kmeans.cluster_centers_)

    return(pred_labels)

def test_func():
    X_train = []
    for i in range(5):
        X_train.append(create_descriptor(fake_dists()))

    X_test = []
    for i in range(20):
        X_test.append(create_descriptor(fake_dists()))

    return(predict_labels(X_train, X_test))

# print(test_func())

    # Fit K neighbors classifier with the histograms and the labels of the training data.
    # neighbors = KNeighborsClassifier().fit(histograms, train_gs[0])





    # Load in testing data, build histograms for the testing data and then use k means classifier to
    # predict on the testing histograms. This will yield label predictions for every test image.

    # iterate over all test images and create a histogram for each image
    # X_test = test_D
    # histograms_test = []
    # for i in range(len(X_test)):
    #     image = X_test[i]
        
    #     # use NN to assign each SIFT descriptor to a centroid (build histogram)
    #     distance, indices = tree.query(image, k=1)     
    #     histogram = np.histogram(indices, bins=np.arange(1, 122))
    #     histogram = histogram[0].reshape(1,-1)
    #     histograms_test.append(histogram)




    # labels = []
    # for i in range(len(histograms_test)):
    #     labels.append(neighbors.predict(histograms_test[i]))





    # matrix = confusion_matrix(test_gs[0], labels)
    # display = ConfusionMatrixDisplay(confusion_matrix=matrix, display_labels=neighbors.classes_)
    # display.plot()
    # plt.show()