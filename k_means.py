import pandas as pd
import sys
import numpy as np
import matplotlib.pyplot as pyplot
import heapq
from sklearn import cluster

#This script will take in csv accelerometer data, store each axis (x,y,z) as a numpy array and then run the kmeans clustering algorithm against the axis specified
#Script arguments are file, k, and axis

shots = pd.read_csv(sys.argv[1])
print shots.head()

index = shots.loc[0:].index

x = np.array(index)

x_values = shots.loc[0:]['x']
y_values = shots.loc[0:]['y']
z_values = shots.loc[0:]['z']

y_xs = np.array(x_values)
y_ys = np.array(y_values)
y_zs = np.array(z_values)

data = np.column_stack((x,y_xs))
print data

k = sys.argv[2]
k = int(k)
kmeans = cluster.KMeans(n_clusters=k, max_iter=1000)
kmeans.fit(data)
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
print "KMEAN CENTROIDS:"
print centroids
for i in range(k):
    # select only data observations with cluster label == i
    ds = data[np.where(labels==i)]
    # plot the data observations
    pyplot.plot(ds[:,0],ds[:,1],'o')
    # plot the centroids
    lines = pyplot.plot(centroids[i,0],centroids[i,1],'kx')
    # make the centroid x's bigger
    pyplot.setp(lines,ms=15.0)
    pyplot.setp(lines,mew=2.0)
pyplot.show()
