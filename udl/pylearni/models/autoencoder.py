import theano
from theano import tensor

import pylearn2
from pylearn2.corruption import BinomialCorruptor
import pylearn2.models.autoencoder
from udl.model import UDLModel
from udl.pylearni.datasets.VectorDataset import VectorDataset
from udl.pylearni.trainers.SGDTrainer import SGDTrainer

__author__ = 'haitham'


def abstractfit(model, X_train):
    nfeat = X_train.shape[1]
    model.configs['nvis'] = nfeat
    dataset = model.dataset_adaptor.fit_transform(X_train)
    super(type(model), model).__init__(**model.configs)
    trainer = model.trainer.get_trainer(model, dataset)
    trainer.main_loop()
    # define estimator
    X = tensor.matrix()
    # ff = theano.function([X], model.encode(X), compile.mode.Mode(linker='py', optimizer='fast_compile'))
    ff = theano.function([X], model.encode(X))

    # print ff(X_train)
    model.estimator = ff
    return model


class Autoencoder(pylearn2.models.autoencoder.Autoencoder, UDLModel):
    def __init__(self, nvis=0, nhid=10, act_enc='sigmoid', act_dec='sigmoid', tied_weights=False, irange=1e-3, rng=9001,
                 dataset_adaptor=VectorDataset(),
                 trainer=SGDTrainer()):
        self.configs = {'nvis': nvis,
                        'nhid': nhid,
                        'act_enc': act_enc,
                        'act_dec': act_dec,
                        'tied_weights': tied_weights,
                        'irange': irange,
                        'rng': rng,
                        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

    def fit(self, X_train):
        return abstractfit(self, X_train)


class DenoisingAutoencoder(pylearn2.models.autoencoder.DenoisingAutoencoder, UDLModel):
    def __init__(self,
                 corruptor=BinomialCorruptor(0.5),
                 nvis=0,
                 nhid=10,
                 act_enc='sigmoid',
                 act_dec='sigmoid',
                 tied_weights=False,
                 irange=1e-3,
                 rng=9001,
                 dataset_adaptor=VectorDataset(),
                 trainer=SGDTrainer()
                 ):
        self.configs = {'corruptor': corruptor,
                        'nvis': nvis,
                        'nhid': nhid,
                        'act_enc': act_enc,
                        'act_dec': act_dec,
                        'tied_weights': tied_weights,
                        'irange': irange,
                        'rng': rng,
                        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

    def fit(self, X_train):
        return abstractfit(self, X_train)


class ContractiveAutoencoder(pylearn2.models.autoencoder.ContractiveAutoencoder, UDLModel):
    def __init__(self,
                 nvis=0,
                 nhid=10,
                 act_enc='sigmoid',
                 act_dec='sigmoid',
                 tied_weights=False,
                 irange=1e-3,
                 rng=9001,
                 dataset_adaptor=VectorDataset(),
                 trainer=SGDTrainer()):
        self.configs = {
            'nvis': nvis,
            'nhid': nhid,
            'act_enc': act_enc,
            'act_dec': act_dec,
            'tied_weights': tied_weights,
            'irange': irange,
            'rng': rng,
        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

    def fit(self, X_train):
        return abstractfit(self, X_train)


class HigherOrderContractiveAutoencoder(pylearn2.models.autoencoder.HigherOrderContractiveAutoencoder, UDLModel):
    def __init__(self, corruptor=BinomialCorruptor(0.5),
                 num_corruptions=2,
                 nvis=0,
                 nhid=10,
                 act_enc='sigmoid',
                 act_dec='sigmoid',
                 tied_weights=False,
                 irange=1e-3,
                 rng=9001,
                 dataset_adaptor=VectorDataset(),
                 trainer=SGDTrainer()):
        self.configs = {'corruptor': corruptor,
                        'num_corruptions': num_corruptions,
                        'nvis': nvis,
                        'nhid': nhid,
                        'act_enc': act_enc,
                        'act_dec': act_dec,
                        'tied_weights': tied_weights,
                        'irange': irange,
                        'rng': rng,
                        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

    def fit(self, X_train):
        return abstractfit(self, X_train)

# class UntiedAutoencoder(Autoencoder):
#
# # class DeepComposedAutoencoder(AbstractAutoencoder):
