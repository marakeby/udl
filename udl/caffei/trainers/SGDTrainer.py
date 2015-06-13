import tempfile
import types

import caffe
import udl.caffei.models.caffe_pb2 as pb2
from udl.caffei.models.layers.layer import set

from udl.caffei.models.utils import get4D

__author__ = 'haitham'



def set_train_data(self, X, Y):
    # TODO find better way to assign data, support batches
    X4D = get4D(X)
    Y4D = get4D(Y)
    sh = Y4D.shape
    self.net.blobs['label'].reshape (sh[0], sh[1], sh[2], sh[3])
    self.net.blobs['data'].data[...] = X4D
    self.net.blobs['label'].data[...] = Y4D

def set_test_data(self, X):
    X4D= get4D(X)
    sh = X4D.shape
    self.test_nets[0].blobs['data'].reshape (sh[0], sh[1], sh[2], sh[3])
    self.test_nets[0].blobs['data'].data[...] = X4D

class SGDTrainer():
    def __init__(self):
        pass

    def get_trainer(self, model, X =None, y = None):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(model)
        f.close()

        self.params = pb2.SolverParameter()
        self.params.net = f.name

        set(self.params.test_iter, 10)
        self.params.test_interval = 10
        self.params.base_lr = 0.01
        self.params.momentum = 0.9
        self.params.weight_decay = 0.0005
        self.params.lr_policy = 'inv'
        self.params.gamma = 0.0001
        self.params.power = 0.75
        self.params.display = 100
        self.params.max_iter = 100
        self.params.snapshot_after_train = False


        f2 = tempfile.NamedTemporaryFile(delete=False)
        f2.write(self.params.__str__())
        f2.close()

        self.solver = caffe.SGDSolver(f2.name)
        self.solver.set_train_data = types.MethodType(set_train_data, self.solver)
        self.solver.set_test_data = types.MethodType(set_test_data, self.solver)

        return self.solver
