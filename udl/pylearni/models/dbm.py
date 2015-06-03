__author__ = 'haitham'
import theano
from theano import tensor

from udl.model import UDLModel
import pylearn2
from udl.pylearni.datasets.VectorDataset import VectorDataset
from udl.pylearni.trainers.SGDTrainer import SGDTrainer
from pylearn2.costs.dbm import VariationalPCD, TorontoSparsity, WeightDecay


class DBM(pylearn2.models.dbm.dbm.DBM, UDLModel):
    """
    A deep Boltzmann machine.

    See "Deep Boltzmann Machines" by Ruslan Salakhutdinov and Geoffrey Hinton
    for details.

    Pylearn2 Parameters
    ----------
    batch_size : int
        The batch size the model should use. Some convolutional
        LinearTransforms require a compile-time hardcoded batch size,
        otherwise this would not be part of the model specification.
    visible_layer : VisibleLayer
        The visible layer of the DBM.
    hidden_layers : list
        The hidden layers. A list of HiddenLayer objects. The first
        layer in the list is connected to the visible layer.
    niter : int
        Number of mean field iterations for variational inference
        for the positive phase.
    sampling_procedure : SamplingProcedure (optional)
        An object that specifies how to draw samples from the model.
        If not specified, some standard algorithm will be used.
    inference_procedure : InferenceProcedure (optional)
        An object that specifies how to perform mean field inference
        in the model. If not specified, some standard algorithm will
        be used.


    UDL Parameters
    ----------
    dataset_adaptor
    trainer
    TODO: WRITEME
    """

    def __init__(self, batch_size=10, visible_layer=None,
                 hidden_layers=None,  # if it is non, set it in __init()__
                 niter=2,
                 sampling_procedure=None,
                 inference_procedure=None,
                 nhid=100,
                 dataset_adaptor=VectorDataset(),
                 trainer=None  # if it is non, set it in __init()__
                 ):

        if hidden_layers is None:
            hidden_layers = [
                pylearn2.models.dbm.BinaryVectorMaxPool(detector_layer_dim=nhid, layer_name='h1', irange=.05,
                                                        pool_size=1),
                pylearn2.models.dbm.BinaryVectorMaxPool(detector_layer_dim=nhid / 2, layer_name='h2', irange=.05,
                                                        pool_size=1)]
        if trainer is None:
            trainer = SGDTrainer(pylearn2.costs.cost.SumOfCosts(
                [VariationalPCD(num_chains=100, num_gibbs_steps=5), WeightDecay(coeffs=[.0001, .0001]),
                 TorontoSparsity(targets=[.2, .2],
                                 coeffs=[.001, .001], )]))

        # use configs to store all the configurable parameters of the model, it will be used to configure the parent class
        self.configs = {'batch_size': batch_size,
                        'hidden_layers': hidden_layers,
                        'visible_layer': visible_layer,
                        'sampling_procedure': sampling_procedure,
                        'inference_procedure': inference_procedure,
                        'niter': niter,
                        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

        return

    def fit(self, X_train, **kwargs):
        nfeat = X_train.shape[1]
        dataset = self.dataset_adaptor.fit_transform(X_train)
        self.configs['visible_layer'] = pylearn2.models.dbm.BinaryVector(nvis=nfeat)
        print self.configs
        super(DBM, self).__init__(**self.configs)
        trainer = self.trainer.get_trainer(self, dataset)
        trainer.main_loop()
        X = tensor.matrix()
        estimator = theano.function([X], self.reconstruct(X))
        self.estimator = estimator
        return self


class RBM(DBM):
    """
    A Restricted Boltzmann machine.



    Pylearn2 Parameters (copied from pylearn2)
    ----------
    batch_size : int
        The batch size the model should use. Some convolutional
        LinearTransforms require a compile-time hardcoded batch size,
        otherwise this would not be part of the model specification.
    visible_layer : VisibleLayer
        The visible layer of the DBM.
    hidden_layers : list
        The hidden layers. A list of HiddenLayer objects. The first
        layer in the list is connected to the visible layer.
    niter : int
        Number of mean field iterations for variational inference
        for the positive phase.
    sampling_procedure : SamplingProcedure (optional)
        An object that specifies how to draw samples from the model.
        If not specified, some standard algorithm will be used.
    inference_procedure : InferenceProcedure (optional)
        An object that specifies how to perform mean field inference
        in the model. If not specified, some standard algorithm will
        be used.


    UDL Parameters
    ----------
    dataset_adaptor
    trainer
    TODO: WRITEME

    Notes (copied from pylearn2)
    -----
    The `RBM` class is redundant now that we have a `DBM` class, since
    an RBM is just a DBM with one hidden layer. Users of pylearn2 should
    use single-layer DBMs when possible. Not all RBM functionality has
    been ported to the DBM framework yet, so this is not always possible.
    (Examples: spike-and-slab RBMs, score matching, denoising score matching)
    pylearn2 developers should not add new features to the RBM class or
    add new RBM subclasses. pylearn2 developers should only add documentation
    and bug fixes to the RBM class and subclasses. pylearn2 developers should
    finish porting all RBM functionality to the DBM framework, then turn
    the RBM class into a thin wrapper around the DBM class that allocates
    a single layer DBM.

    """

    def __init__(self, batch_size=10, visible_layer=None,
                 niter=2,
                 sampling_procedure=None,
                 inference_procedure=None,
                 nhid=100,
                 dataset_adaptor=VectorDataset(),
                 trainer=None  # if it is non, set it in __init()__
                 ):
        hidden_layers = [
            pylearn2.models.dbm.BinaryVectorMaxPool(detector_layer_dim=nhid, layer_name='h', irange=.05, pool_size=1)]
        if trainer is None:
            trainer = SGDTrainer(pylearn2.costs.cost.SumOfCosts(
                [VariationalPCD(num_chains=100, num_gibbs_steps=5), WeightDecay(coeffs=[.0001]),
                 TorontoSparsity(targets=[.2],
                                 coeffs=[.001], )]))

        # use configs to store all the configurable parameters of the model, it will be used to configure the parent class
        self.configs = {'batch_size': batch_size,
                        'hidden_layers': hidden_layers,
                        'visible_layer': visible_layer,
                        'sampling_procedure': sampling_procedure,
                        'inference_procedure': inference_procedure,
                        'niter': niter,
                        }

        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer

        return
