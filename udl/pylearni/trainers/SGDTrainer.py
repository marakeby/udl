__author__ = 'marakeby'
from pylearn2.training_algorithms.sgd import SGD
from pylearn2.costs.autoencoder import MeanSquaredReconstructionError
from pylearn2.termination_criteria import EpochCounter
from pylearn2.train import Train


class SGDTrainer():
    def __init__(self, cost=MeanSquaredReconstructionError()):
        self.cost = cost
        self.train_iteration_mode = 'even_sequential'
        self.monitor_iteration_mode = 'even_sequential'

    def get_trainer(self, model, trainset):
        MAX_EPOCHS_UNSUPERVISED = 100
        # configs on sgd
        train_algo = SGD(
            learning_rate=0.1,
            # cost =  MeanSquaredReconstructionError(),
            cost=self.cost,
            batch_size=10,
            monitoring_batches=10,
            monitoring_dataset=trainset,
            monitor_iteration_mode='even_sequential',
            train_iteration_mode='even_sequential',
            termination_criterion=EpochCounter(max_epochs=MAX_EPOCHS_UNSUPERVISED),
            update_callbacks=None
        )

        extensions = None
        return Train(model=model,
                     algorithm=train_algo,
                     extensions=extensions,
                     dataset=trainset)
