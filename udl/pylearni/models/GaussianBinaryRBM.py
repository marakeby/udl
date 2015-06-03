from pylearn2.corruption import GaussianCorruptor

__author__ = 'marakeby'
from pylearn2.models.rbm import GaussianBinaryRBM
from pylearn2.energy_functions.rbm_energy import GRBM_Type_1
import theano, pylearn2
import sklearn
from theano import tensor
from udl.pylearni.datasets.VectorDataset import VectorDataset
from udl.pylearni.trainers.SGDTrainer import SGDTrainer
from pylearn2.costs.ebm_estimation import SMD


class GaussianBinaryRBM(pylearn2.models.rbm.GaussianBinaryRBM, sklearn.base.BaseEstimator):
    def __init__(self,
                 energy_function_class=GRBM_Type_1,
                 nhid=10,
                 irange=0.5,
                 rng=None,
                 mean_vis=False,
                 init_sigma=.4,
                 learn_sigma=True,
                 sigma_lr_scale=1.,
                 init_bias_hid=-2.,
                 min_sigma=.1,
                 max_sigma=10.,
                 dataset_adaptor=VectorDataset(),
                 trainer=SGDTrainer(SMD(corruptor=GaussianCorruptor(stdev=0.00)))):
        # trainer = SGDTrainer(SMD() )):

        self.config = {
            'energy_function_class': energy_function_class,
            'nhid': nhid,
            'irange': irange,
            'rng': rng,
            'mean_vis': mean_vis,
            'init_sigma': init_sigma,
            'learn_sigma': True,
            'sigma_lr_scale': sigma_lr_scale,
            'init_bias_hid': init_bias_hid,
            'min_sigma': min_sigma,
            'max_sigma': max_sigma,
            'learn_sigma': learn_sigma}

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

        return

    def fit(self, X_train):
        nfeat = X_train.shape[1]

        dataset = self.dataset_adaptor.fit_transform(X_train)
        self.config['nvis'] = nfeat
        print self.config
        super(GaussianBinaryRBM, self).__init__(**self.config)

        trainer = self.trainer.get_trainer(self, dataset)
        trainer.main_loop()
        model = trainer.model

        X = tensor.matrix()
        ff = theano.function([X], model.P_H_given_V(X))

        self.model = ff
        return self

    def predict(self, X_test):
        pred = self.model(X_test)
        return pred

    def transform(self, X_test, y=None, **fit_params):
        pred = self.model(X_test)
        return pred

    def fit_transform(self, X, y=None, **fit_params):
        self.fit(X)
        return self.transform(X)
