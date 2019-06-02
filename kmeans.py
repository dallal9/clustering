# sklearn.cluster.KMeans
#https: // scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
from sklearn.cluster import KMeans
import numpy as np
import time 
from sklearn import metrics


X = np.array([[1, 2], [1, 4], [1, 0],[10, 2], [10, 4], [10, 0]])
Y = np.array([0,0,0,1,1,1])

'''parameters
max: means no certian limit on the value
'''

n_clusters = 3     # int, optional, default: 8 - The number of clusters
                    # int => [2, number of samples]  

init = 'k-means++' # Method for initialization, defaults to ‘k-means++
                    # ['k-means++','random', ndarray]

n_init = 10        # int, default: 10 - Number of time the k-means algorithm will be run with different centroid seeds
                    # int => [1,max] - directly increase processing time 

max_iter = 300  # int, default: 300 - Maximum number of iterations of the k-means algorithm for a single run.
                    # int => [1,max]

tol = 0.0001  #float, default: 1e-4
                #any float value [-max,max]

precompute_distances = 'auto'  #Precompute distances(faster but takes more memory).
                                # ['auto', True, False}
                                #‘auto’: do not precompute distances if n_samples * n_clusters > 12 million. 

random_state = 0  # int, RandomState instance or None (default)
                      #Determines random number generation for centroid initialization. Use an int to make the randomness deterministic.

copy_x = True  #When pre-computing distances it is more numerically accurate to center the data first. If copy_x is True (default), 
                #then the original data is not modified, ensuring X is C-contiguous. If False, the original data is modified, and put 
                # back before the function returns, but small numerical differences may be introduced
                #True or False



n_jobs = None  #int or None 
                #The number of jobs to use for the computation. This works by computing each of the n_init runs in parallel.

algorithm = 'auto' # “auto”, “full” or “elkan”, default=”auto”
                    #“auto” chooses “elkan” for dense data and “full” for sparse data.
verbose = 0

'''
model
'''
t1 = time.time()
kmeans = KMeans(n_clusters=n_clusters, init=init, n_init=n_init, max_iter=max_iter,
                tol=tol, precompute_distances=precompute_distances, random_state=random_state,
                copy_x=copy_x, n_jobs=n_jobs, algorithm=algorithm,verbose=verbose).fit(X)
print(time.time()-t1)
kmeans.labels_

kmeans.predict([[0, 0], [12, 3]])
y0 = kmeans.predict([[1, 2], [1, 4], [1, 0], [10, 2], [10, 4], [10, 0]])
kmeans.cluster_centers_


'''evaluations
'''
print(kmeans.inertia_)  # + no need for Y, - makes the assumption that clusters are convex and isotropic, -  not a normalized metric

print(metrics.adjusted_rand_score(Y, y0))  # - need Y , + Bounded range [-1, 1], + No assumption is made on the cluster structure

print(metrics.adjusted_mutual_info_score(Y, y0))  # - need Y,+Upper bound 1,  +Random (uniform)

##
#homogeneity: each cluster contains only members of a single class.
#completeness: all members of a given class are assigned to the same cluster.
#v = (1 + beta) * homogeneity * completeness/ (beta * homogeneity + completeness)
##
# + Bounded range [0.0, 1.0], + No assumption is made on the cluster structure, +Intuitive interpretation
#- not normalized with regards to random labeling, - For smaller sample sizes or larger number of clusters it is safer to use an adjusted index such as the Adjusted Rand Index (ARI).

print(metrics.homogeneity_score(Y, y0))
print(metrics.completeness_score(Y, y0))
print(metrics.v_measure_score(Y, y0))

print(metrics.fowlkes_mallows_score(Y, y0))
