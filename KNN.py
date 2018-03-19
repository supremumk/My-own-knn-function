import numpy as np
import pandas
import time

# seed for random functions
SEED = 57553807

# percent to take in training set
SPLIT_PCT = 0.8

def split(x, y):
    """
    Split the dataset into subtrain + validation sets
    """
    assert len(x) == len(y)
        
    # randomly permute indices to determine train / val split
    np.random.seed(SEED)
    indices = np.random.permutation(range(len(x)))
    split_at = int(SPLIT_PCT * len(indices))
    train_ind, val_ind = indices[:split_at], indices[split_at:]

    # split the dataset using train_ind, val_ind
    return x[train_ind], y[train_ind], x[val_ind], y[val_ind]
    
def standardize(xtrain, xtest):
    """
    Standardize xtrain, xtest by subtracting mean and dividing by std of xtrain
    """
    # calculate mean, std over training set
    mean, std = xtrain.mean(), xtrain.std()
    
    # subtract mean, divide by std
    xtrain_st = (xtrain - mean) / std
    xtest_st = (xtest - mean) / std
    
    return xtrain_st, xtest_st
    
def dist_matrix(xtrain, xtest):
    """
    Each row contains distances from the corresponding xtest_row to all xtrain_rows.
    Rows iterate through xtest, while the columns iterate through different xtrain rows.
    """
    start = time.time()
    
    x, y = xtrain, xtest
    
    # square xtrain and stretch (tile) vertically
    xx = np.tile(np.sum(x**2, axis=1), (len(y), 1))
    
    # square xtest and stretch (tile) horizontally
    yy = np.tile(np.sum(y**2, axis=1).reshape((-1, 1)), (1, len(x)))
    
    # compute x * y dot products
    xy = np.matmul(y, x.T)
    
    # euclidean dist: (x - y) ^ 2 = xx - 2xy + yy
    matrix = xx - 2 * xy + yy
    
    print('computed distance matrix ({},{}) in {} s'.format(*matrix.shape, time.time() - start))
    return matrix

def knnclass_k(dist_matrix, ytrain, k):
    """
    Grunt work behind KNN classification (after calculating distance matrix).
    Predict most frequently occurring label.
    """
    
    predictions = []
    for xtest_to_xtrain in dist_matrix:

        # find the ytrain values corresponding to k-smallest distances
        k_smallest = np.argpartition(xtest_to_xtrain, k)[:k]

        y_neighborhood = [
            ytrain[idx] for idx in k_smallest
        ]

        # predict the most frequently occurring label
        predictions.append(max(set(y_neighborhood), key=y_neighborhood.count))

    return predictions

def misclassification_error(yhat, y):
    """
    Calculate misclassification error as # wrong / total # predictions
    """
    wrong = sum(int(pred != actual) for pred, actual in zip(yhat, y))
    return wrong / float(len(y))

def knnclass(xtrain, xtest, ytrain):
    """
    Perform KNN classification on xtrain, xtest, ytrain.
    """
    print('## START KNN CLASSIFICATION ##')
    
    # convert from pandas.DataFrame to numpy arrays
    xtrain, xtest, ytrain = xtrain.values, xtest.values, ytrain.values
    
    # split into subtrain, validation sets
    xsubtrain, ysubtrain, xval, yval = split(xtrain, ytrain)

    # compute the distance matrix once!
    errs = {}
    dist = dist_matrix(*standardize(xsubtrain, xval))
    
    # try values of k from 2 -> 15
    for k in range(2, 15):
        # make predictions, record misclassification error
        predictions = knnclass_k(dist, ysubtrain, k)
        errs[k] = misclassification_error(predictions, yval)
        print('k={} \tyields \terr={}'.format(k, errs[k]))
    
    best_k = min(errs.items(), key=lambda item: item[1])[0]
    
    # return predictions using the best k on the entire training dataset
    return pandas.DataFrame({
        'knn_pred': knnclass_k(dist_matrix(*standardize(xtrain, xtest)), ytrain, best_k)
    })
