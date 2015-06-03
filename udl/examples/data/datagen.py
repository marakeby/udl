__author__ = 'haitham'

import sklearn
import sklearn.datasets
import sklearn.linear_model


def get_sklearn_data():
    X, Y = sklearn.datasets.make_classification(
        n_samples=10000, n_features=4, n_redundant=0, n_informative=2,
        n_clusters_per_class=2, hypercube=False, random_state=0
    )

    # Split into train and test
    X, Xt, Y, Yt = sklearn.cross_validation.train_test_split(X, Y)
    return X, Xt, Y, Yt
