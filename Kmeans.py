import numpy as np
import math

########################################################################
#######  you should maintain the  return type in starter codes   #######
########################################################################

def update_assignments(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  # Output:
  #   a is the cluster assignments (n,), 1-d array

    a = np.zeros(X.shape[0])
    n = len(X)
    d = len(X[0])
    k = len(C)

    for i in range(n):
        v_1 = X[i]
        min_d = -1
        for j in range(k):
            di = 0
            v_2 = C[j]
            for l in range(d):
                di += (v_1[l] - v_2[l]) ** 2
            if (min_d == -1):
                min_d = di
                a[i] = j
            if (di < min_d):
                min_d = di
                a[i] = j
    return a

def update_centers(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   C is the new cluster centers (k, d), 2-d array
    n = len(X)
    d = len(X[0])
    k = len(C)
    for i in range(k):
        l = []
        for j in range(n):
            if (a[j] == i):
                l.append(X[j])
        for p in range(d):
            di = 0.0
            for q in range(len(l)):
                di += l[q][p]
            if (len(l) != 0):
                C[i][p] = (di * 1.0) / (len(l) * 1.0)

    return C

#X = np.array([[2.0,2.0],[1.0,2.0],[4.0,5.0],[2.0,3.0],[4.1,4.0],[5.1,4.0]])
#C = np.array([[3,5],[9,5]])
#a = np.array([0,0,1,0,1,1])

#print(update_centers(X,C,a))

def arr_compare(a, b):
    m = len(a)
    for i in range(m):
        if a[i] != b[i]:
            return True
    return False

def lloyd_iteration(X, C):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the initial cluster centers (k, d), 2-d array
  # Output:
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
    a = update_assignments(X, C)
    C = update_centers(X, C, a)
    a_prime = update_assignments(X, C)
    while (arr_compare(a, a_prime)):
        C = update_centers(X, C, a_prime)
        tmp = update_assignments(X, C)
        a = a_prime
        a_prime = tmp
    return (C, a)
#print(lloyd_iteration(X, C))

def kmeans_obj(X, C, a):
  # Input:
  #   X is the data matrix (n, d),  2-d array
  #   C is the cluster centers (k, d), 2-d array
  #   a is the cluster assignments (n,), 1-d array
  # Output:
  #   obj is the k-means objective of the provided clustering, scalar, float
    n = len(X)
    d = len(X[0])
    k = len(C)
    obj = 0.0
    for i in range(n):
       for x in range(d):
            obj += (X[i][x] - C[a[i]][x]) ** 2

    return obj


########################################################################
#######          DO NOT MODIFY, BUT YOU SHOULD UNDERSTAND        #######
########################################################################

# kmeans_cluster will be used in the experiments, it is available after you 
# have implemented lloyd_iteration and kmeans_obj.

def kmeans_cluster(X, k, init, num_restarts):
  n = X.shape[0]
  # Variables for keeping track of the best clustering so far
  best_C = None
  best_a = None
  best_obj = np.inf
  for i in range(num_restarts):
    if init == "random":
      perm = np.random.permutation(range(n))
      C = np.copy(X[perm[0:k]])
    elif init == "kmeans++":
      C = kmpp_init(X, k)
    elif init == "fixed":
      C = np.copy(X[0:k])
    else:
      print "No such module"
    # Run the Lloyd iteration until convergence
    (C, a) = lloyd_iteration(X, C)
    # Compute the objective value
    obj = kmeans_obj(X, C, a)
    if obj < best_obj:
      best_C = C
      best_a = a
      best_obj = obj
  return (best_C, best_a, best_obj)



########################################################################
#######     DO NOT MODIFY, READ ONLY IF YOU ARE INTERESTED       #######
########################################################################

def kmpp_init(X, k):
  n = X.shape[0]
  sq_distances = np.ones(n)
  center_ixs = list()
  for j in range(k):
    # Choose a new center index using D^2 weighting
    ix = discrete_sample(sq_distances)
    # Update the squared distances for all points
    deltas = X - X[ix]
    for i in range(n):
      sq_dist_to_ix = np.power(np.linalg.norm(deltas[i], 2), 2)
      sq_distances[i] = min(sq_distances[i], sq_dist_to_ix)
    # Append this center to the list of centers
    center_ixs.append(ix)
  # Output the chosen centers
  C = X[center_ixs]
  return np.copy(C)


def discrete_sample(weights):
  total = np.sum(weights)
  t = np.random.rand() * total
  p = 0.0
  for i in range(len(weights)):
    p = p + weights[i];
    if p > t:
      ix = i
      break
  return ix