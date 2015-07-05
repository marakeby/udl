import tempfile
import types

import caffe
import udl.caffei.models.caffe_pb2 as pb2
from udl.caffei.models.layers.layer import set

from udl.caffei.models.utils import get4D
import numpy as np
__author__ = 'haitham'



# def set_train_data(self, X, Y):
#     # TODO find better way to assign data, support batches
#     # self.net.set_input_arrays(X,Y)
#     X4D = get4D(X)
#     Y4D = get4D(Y)
#     sh = Y4D.shape
#     self.net.blobs['label'].reshape (sh[0], sh[1], sh[2], sh[3])
#     self.net.blobs['data'].data[...] = X4D
#     self.net.blobs['label'].data[...] = Y4D
#
# def set_test_data(self, X):
#     X4D= get4D(X)
#     sh = X4D.shape
#     self.test_nets[0].blobs['data'].reshape (sh[0], sh[1], sh[2], sh[3])
#     self.test_nets[0].blobs['data'].data[...] = X4D




class SGDTrainer():
    def __init__(self):
        pass

    def create_batches(self, ):
        def chunks(l, n):
            for i in xrange(0, l.shape[0], n):
                yield l[i:i+n]

        def shuffle_data(x, y):
            state = np.random.get_state()
            np.random.shuffle(x)
            np.random.set_state(state)
            np.random.shuffle(y)
            return x,y

        # np.random.shuffle(self.X)
        self.X , self.y =shuffle_data(self.X, self.y)
        batch_size = self.solver.net.blobs['data'].num
        self.x_batches = list(chunks(self.X, batch_size))
        self.y_batches = list(chunks(self.y, batch_size))

    def solve(self):
        l= len( self.x_batches)
        for i in range(self.params.max_iter):
            idx = i%l
            x_batch = self.x_batches[idx]
            y_batch = self.y_batches[idx]

            self.set_train_data( x_batch,y_batch)
            self.solver.step(1)

    def init_trainer(self, model, X =None, y = None):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(model)
        f.close()
        self.X=X
        self.y =y

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
        self.params.display = 10
        self.params.max_iter = 100
        self.params.snapshot_after_train = False


        f2 = tempfile.NamedTemporaryFile(delete=False)
        f2.write(self.params.__str__())
        f2.close()

        self.solver = caffe.SGDSolver(f2.name)
        f2.delete = True
        if (X is not None) and (y is not None):
            self.create_batches()
        # self.solver.set_train_data = types.MethodType(set_train_data, self.solver)
        # self.solver.set_test_data = types.MethodType(set_test_data, self.solver)

        return self.solver

    def set_train_data(self, X, Y):
        # TODO find better way to assign data, support batches
        # self.net.set_input_arrays(X,Y)
        X4D = get4D(X)
        Y4D = get4D(Y)
        sh = Y4D.shape
        self.solver.net.blobs['label'].reshape (sh[0], sh[1], sh[2], sh[3])
        self.solver.net.blobs['data'].data[...] = X4D
        self.solver.net.blobs['label'].data[...] = Y4D

    def set_test_data(self, X):
        X4D= get4D(X)
        sh = X4D.shape
        self.solver.test_nets[0].blobs['data'].reshape (sh[0], sh[1], sh[2], sh[3])
        self.solver.test_nets[0].blobs['data'].data[...] = X4D
    def predict(self, x_test):
        X4D = get4D(x_test)
        self.set_test_data (X4D)
        self.solver.test_nets[0].forward()