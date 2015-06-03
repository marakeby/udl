__author__ = 'haitham'
import sklearn


class UDLModel(sklearn.base.BaseEstimator):
    # self.estimator =None
    # self.configs= {}
    def __init__(self):
        # a dictionary of all configurable parameters. This dictionary will be used in get_params. see the note in get_params function
        self.configs = {}
        # there is no restriction on the internal implementation of you estimator. You estimator has to be able to transform input into output, see predict function
        self.estimator = None

    def fit(self, x_train, **kwargs):
        # fit() is model dependent
        raise NotImplementedError

    def predict(self, x_test):
        assert self.estimator
        pred = self.estimator(x_test)
        return pred

    def transform(self, x_test, y=None, **fit_params):
        pred = self.estimator(x_test)
        return pred

    def fit_transform(self, x, y=None, **fit_params):
        self.fit(x)
        return self.transform(x)

    def get_params(self, deep=True):
        # Note: Pylearn2 has a function with the same name, please replace with another name e.g. get_params_list
        return self.configs

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            self.setattr(parameter, value)
            #     return self
