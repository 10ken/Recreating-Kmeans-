########## K means (++) ##########

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

df = pd.read_csv("customer.csv", header=1)
gender = {'Male': 1,'Female': 0}

df.Gender = [gender[item] for item in df.Gender]

df.head()

def kmeans(features, k, p=False, rs = 100):
    """ Return k cluters given df features, rs is random_state it initalize mean vector
    """
    def kmeans_plus(features, k):
        """ Returns ndarray
        initialization of mean vector given kmeans++
        """
        # Repeat until k centers have been chosen.
        # Choose one center uniformly at random from among the data points.
        x = np.array(features.sample(1, random_state=rs))[0]
        means = [x]
        # since inital weight, only need k-1 clusters
        for i in range(k-1):
    
            # For each data point x, compute D(x), the distance between x and the nearest center 
            # that has already been chosen.
            D2 = []
            for idx, row in features.iterrows():
                D2.append(min([np.linalg.norm(row.to_numpy() - mean)**2 for mean in means]))
            # Choose one new data point at random as a new center, using a weighted 
            # probability distribution where a point x is chosen with probability proportional to D(x)**2.
            x = np.array(features.sample(1, weights=D2, random_state=rs))[0]
            means.append(x)
        return np.array(means)
    
    # initialize clusters 1-k
    S = dict()
    for i in range(k):
        S[i] = []
    
    # initalize mean vector
    if p:
        means = kmeans_plus(features, k)
        
    else:
        means = np.array(features.sample(k, random_state=rs))
    # Perform until convergence
    prev_means = np.copy(means)
    while True:
        # add to min cluster
        for idx, row in features.iterrows():
            m = [np.linalg.norm(row-mean)**2 for mean in means]
            i = m.index(min(m))
            S[i].append(row)
        
        # Refactor mean array
        for i in range(k):
            if len(S[i]):
                means[i] = sum(S[i])/len(S[i])
            else:
                means[i] = sum(S[i])
        
        # converged
        if np.array_equal(prev_means, means):
            break
        else:
            for i in range(k):
                S[i] = []
            prev_means = np.copy(means)
    return S


def my_kmeans_plot(clusters):
    """plots the clusters
    """
    colors = 'bgrcmykw'
    fig = plt.figure(figsize=(15,4))
    ### set up first graph
    ax = fig.add_subplot(121)
    for k, c in zip(clusters.keys(), colors):
        ys = [clusters[k][i][1] for i in range(len(clusters[k]))]
        zs = [clusters[k][i][2] for i in range(len(clusters[k]))]
        ax.scatter(ys, zs, c=c)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income')
    
    ### set up second graph
    ax = fig.add_subplot(122, projection='3d')
    for k, c in zip(clusters.keys(), colors):
        xs = [clusters[k][i][0] for i in range(len(clusters[k]))]
        ys = [clusters[k][i][1] for i in range(len(clusters[k]))]
        zs = [clusters[k][i][2] for i in range(len(clusters[k]))]
        ax.scatter(xs, ys, zs, c=c)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Age')
    ax.set_zlabel('Annual Income')
    plt.show()

def my_kmeans_plot(clusters):
    """plots the clusters
    """
    colors = 'bgrcmykw'
    fig = plt.figure(figsize=(15,4))
    ### set up first graph
    ax = fig.add_subplot(121)
    for k, c in zip(clusters.keys(), colors):
        ys = [clusters[k][i][1] for i in range(len(clusters[k]))]
        zs = [clusters[k][i][2] for i in range(len(clusters[k]))]
        ax.scatter(ys, zs, c=c)
    ax.set_xlabel('Age')
    ax.set_ylabel('Annual Income')
    
    ### set up second graph
    ax = fig.add_subplot(122, projection='3d')
    for k, c in zip(clusters.keys(), colors):
        xs = [clusters[k][i][0] for i in range(len(clusters[k]))]
        ys = [clusters[k][i][1] for i in range(len(clusters[k]))]
        zs = [clusters[k][i][2] for i in range(len(clusters[k]))]
        ax.scatter(xs, ys, zs, c=c)
    ax.set_xlabel('Gender')
    ax.set_ylabel('Age')
    ax.set_zlabel('Annual Income')
    plt.show()



features = df[list(df)[1:-1]] # Gender, Age, Income

clus_plus = [kmeans(features, k, True) for k in range(2, 6)]

for c in clus_plus:
    my_kmeans_plot(c)
    
clus = [kmeans(features, k, False) for k in range(2, 6)]

for c in clus:
    my_kmeans_plot(c)