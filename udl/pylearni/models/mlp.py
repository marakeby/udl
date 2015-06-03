# from pylearn2.models.mlp import MLP
from theano import tensor, theano

import pylearn2.models.mlp
from udl.model import UDLModel
from udl.pylearni.datasets.VectorDataset import VectorDataset
from udl.pylearni.trainers.SGDTrainer import SGDTrainer
from pylearn2.costs.mlp import Default

__author__ = 'haitham'


def abstractfit(model, X_train, y):
    nfeat = X_train.shape[1]
    model.configs['nvis'] = nfeat
    dataset = model.dataset_adaptor.fit_transform(X_train, y)
    super(type(model), model).__init__(**model.configs)
    trainer = model.trainer.get_trainer(model, dataset)
    trainer.main_loop()
    # define estimator
    X = tensor.matrix()
    # ff = theano.function([X], model.encode(X), compile.mode.Mode(linker='py', optimizer='fast_compile'))
    ff = theano.function([X], model.fprop(X))

    # print ff(X_train)
    model.estimator = ff
    return model


class MLP(pylearn2.models.mlp.MLP, UDLModel):
    def __init__(self,
                 layers,
                 batch_size=None,
                 input_space=None,
                 input_source='features',
                 target_source='targets',
                 nvis=None,
                 seed=None,
                 layer_name=None,
                 monitor_targets=True,
                 dataset_adaptor=VectorDataset(),
                 trainer=SGDTrainer(cost=Default()),
                 **kwargs):
        self.configs = {'layers': layers,
                        'batch_size': batch_size,
                        'input_space': input_space,
                        'input_source': input_source,
                        'target_source': target_source,
                        'nvis': nvis,
                        'seed': seed,
                        'layer_name': layer_name,
                        'monitor_targets': monitor_targets,
                        # 'kwargs': kwargs,
                        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

    def fit(self, X_train, y):
        return abstractfit(self, X_train, y)
