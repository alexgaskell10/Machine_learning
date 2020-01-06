'''
To do:
-normalize input features
-get target_labels function working properly
'''

import numpy as np
from sklearn.datasets import load_iris
import random
import sys
import itertools

def euclidian_norm(x1,x2):
    norm = sum([(b-a)**2 for a,b in zip(x1,x2)])**0.5
    return norm

def distance_fn(x):
    return 1/(1+x)

def target_labels(target, target_names):
    '''Not working'''
    # Output a dictionary mapping key:flower species
    return {key:name for key,name in zip(target, target_names)}

def knn(x_train, y_train, x, k):
    # Find distance to all other training data obvs and structure data for each obv as a tuple
    nns = []
    for p,q in zip(x_train, y_train):
        d = euclidian_norm(x[:-1], p[:-1])
        nns.append((d,x[-1],q[0]))

    # Find the modal label of the k nearest neighbours
    # Format: (distance, x row_id, n label)
    nns = sorted(nns, key=lambda x: x[0])
    # [print(n) for n in nns[:10]]
    # Return the labels of the k nns
    if distance_flag==True:
        knns = list(np.array(nns)[1:k+1,(0,2)])
        # sort for aggregation
        knns = sorted(knns, key=lambda x: x[1])
        # aggregate to get distance-weighted labels
        knns_agg = []
        for key,rows in itertools.groupby(knns, lambda x: x[1]):
            knns_agg.append((key, sum(distance_fn(r[0]) for r in rows)))
        # return highest distance-weighted label
        knns_label = max(knns_agg, key=lambda x: x[1])[0]
    else:
        knns = list(np.array(nns)[1:k+1,2])
        knns_label = max(set(knns), key=knns.count)
    return knns_label

def main(k, distance_flag):
    # returns accuracy percentage of results
    iris = load_iris()
    n = len(iris['data'])
    # create training data and test data
    random.seed(126)
    # random.seed(random.randint(0,10))
    # create sample selection mask
    mask1 = random.sample(range(n), 2*n//3)
    mask2 = [i for i in range(n) if i not in mask1]
    # add sample id to labels and obvs
    row_id = np.array(list(range(n))).reshape(n,1)
    iris['data'] = np.hstack((iris['data'], row_id))
    iris['target'] = np.hstack((iris['target'].reshape(n,1), row_id))
    ## split into training data and test data
    # x: [0:4] are features, [4] is row_id
    # y: [0] is label, [1] is row_id
    x_train = iris['data'][mask1]
    x_test = iris['data'][mask2]
    y_train = iris['target'][mask1]
    y_test = iris['target'][mask2]

    # Run knn
    ks = {x[-1]:knn(x_train, y_train, x, k) for x in x_test}
    eval = np.array([(f'rowid = {k1}', k == y[0], f'guess: {k}', f'y: {y[0]}')
                        for k,k1,y in zip(ks.values(), ks.keys(), y_test)])
    # Compute accuracy of model
    accuracy = list(eval[:,1]).count('True')*100//len(ks)
    return f'Accuracy = {accuracy}%'

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: python3 filename k [distance_flag=True]')
    else:
        k = int(sys.argv[1])
        if len(sys.argv) == 3:
            distance_flag = sys.argv[2]
        else:
            distance_flag = True
        print(main(k, distance_flag))
