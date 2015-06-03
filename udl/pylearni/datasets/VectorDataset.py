__author__ = 'marakeby'
import numpy as np

import theano

from sklearn.base import TransformerMixin
from pylearn2.datasets.vector_spaces_dataset import VectorSpacesDataset
from pylearn2.space import VectorSpace, CompositeSpace


class VectorDataset(TransformerMixin):
    def fit_transform(self, X, y=None, **fit_params):
        print X.shape
        nfeat = X.shape[1]
        X = X.astype(theano.config.floatX)
        if y is None:
            dataset = VectorSpacesDataset(X, (VectorSpace(nfeat), 'features'))
        else:
            y = np.reshape(y, (y.shape[0], 1))
            space = CompositeSpace([VectorSpace(nfeat), VectorSpace(1)])
            source = ('features', 'targets')
            data_specs = (space, source)
            dataset = VectorSpacesDataset((X, y), data_specs)
        return dataset
