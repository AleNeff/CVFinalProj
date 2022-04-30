import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KDTree, KNeighborsClassifier
from classifier import create_descriptor, get_fake_hand_points

"""
params:
X_train = N 20-d descriptors for ground truth gestures, values in range [0,1] from landmark detector
X_test = M 20-d descriptors for test images, values in range [0,1] from landmark detector

output:
pred_labels = M-d array containing predicted labels of test images

This takes in train and test data (20-d descriptors), creates N clusters, 
and returns the predicted labels of each test image
"""
def build_kmeans(X_train):
    num_clusters = 5
    kmeans = KMeans(n_clusters=num_clusters, random_state=0, max_iter=300).fit(X_train)
    # print(kmeans.labels_)
    # print(pred_labels)
    # print(kmeans.cluster_centers_)
    return(kmeans)

def predict_centroids(X_test, kmodel):
    print(np.array(X_test).shape)
    x = np.expand_dims(X_test, axis=0)
    pred_labels = kmodel.predict(x)
    print(np.array(pred_labels).shape)
    return pred_labels

# def test_func():
#     X_train = []
#     for i in range(5):
#         X_train.append(create_descriptor(fake_dists()))

#     X_test = []
#     for i in range(20):
#         X_test.append(create_descriptor(fake_dists()))

#     return(predict_centroids(X_train, X_test))

"""
Return the center of all of the data points using K-Means. In this case, a 20 dim vector will be returned
"""
def get_cluster_center(train_data):
    kmodel = KMeans(n_clusters=1, max_iter=300).fit(train_data)
    return kmodel.cluster_centers_[0]