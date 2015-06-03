import tempfile

import caffe
import udl.caffei.models.caffe_pb2 as pb2
from udl.caffei.models.layers.layer import set

# from ._caffe import Net, SGDSolver
# from .pycaffe import Net, SGDSolver
# from caffe import get_solver

__author__ = 'haitham'


class SGDTrainer():
    def __init__(self):
        pass

    def get_trainer(self, model, X, y):
        f = tempfile.NamedTemporaryFile(delete=False)
        f.write(model)
        f.close()

        [a, b] = X.shape
        data4D = X.reshape((a, b, 1, 1))
        Y4D = y.reshape((a, 1, 1, 1))
        print data4D.shape

        self.params = pb2.SolverParameter()
        self.params.net = f.name
        # set(params.net_param, model)
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

        # solver = caffe.SGDSolver(params)
        f2 = tempfile.NamedTemporaryFile(delete=False)
        f2.write(self.params.__str__())
        f2.close()

        self.solver = caffe.SGDSolver(f2.name)
        # self.solver = caffe.SGDSolver(self.params.__str__())
        self.solver.net.blobs['data'].data[...] = data4D
        self.solver.net.blobs['label'].data[...] = Y4D
        return self.solver
