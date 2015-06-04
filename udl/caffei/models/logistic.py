import os

import caffe
from udl.caffei.models.layers.data_layers import DummyDataLayer, DummyDataParameter

# from udl.caffei.models.layers.innerproduct import InnerProduct
# from udl.caffei.models.layers.innerproduct import WeightFiller
# from udl.caffei.models.layers.datalayer import DataLayer, DummyDataParams
# from udl.caffei.models.layers.loss import SoftmaxLoss, Softmax
from udl.caffei.models.layers.innerproduct_pb2 import InnerProduct
from udl.caffei.models.layers.layer import Phase
from udl.caffei.models.layers.loss_layers import Loss, LossType
from udl.caffei.models.layers.network import Network
from udl.caffei.trainers.SGDTrainer import SGDTrainer
from udl.model import UDLModel
import numpy as np

class Logistic(UDLModel):
    def __init__(self, ):
        caffe.set_mode_cpu()
        dname = os.path.dirname(os.path.realpath(__file__))
        # self.net_file= os.path.join(dname, 'train_in_memory.prototxt')
        self.net_file = self.get_net()
        # f = self.get_params_file ()
        # self.solver = caffe.SGDSolver(f)

    def fit(self, X, Y):
        model = self.get_net()
        self.solver = SGDTrainer().get_trainer(model, X, Y)
        self.solver.solve()
        return self.solver

    def predict(self, X):
        [a, b] = X.shape
        data4D = X.reshape((a, b, 1, 1))
        # Y4D= Yt.reshape((a, 1, 1,1))
        sh = data4D.shape
        print data4D.shape
        self.solver.test_nets[0].blobs['data'].reshape (sh[0], sh[1], sh[2], sh[3])
        self.solver.test_nets[0].blobs['data'].data[...] = data4D

        # self.solver.test_nets[0].set_input_arrays(data4D, np.random.randint(10,  size=self.solver.test_nets[0].blobs['label'].data.shape))
        self.solver.test_nets[0].forward()



        prob = self.solver.test_nets[0].blobs['output'].data
        print prob.shape
        pred = prob[:, 0] < prob[:, 1]
        return pred

    def get_net(self):
        n = Network()
        n.name = "LogisticRegressionNet"
        d1 = DummyDataLayer(name='data', top=['data', 'label'],
                            params=DummyDataParameter([7500, 7500], [4, 1], [1, 1], [1, 1]))
        # d2 = DummyDataLayer(name='data', top=['data', 'label'], include=Phase.TEST,params = DummyDataParameter([2500,2500],[4,1],[1,1],[1,1]))
        l = InnerProduct(name="fc1", top="fc1", bottom="data", num_output=2)
        l3 = Loss(name="loss", type=LossType.SoftmaxWithLoss, bottom=["fc1", "label"], include=Phase.TRAIN)
        l4 = Loss(name="output", type= LossType.SOFTMAX, top= "output", bottom="fc1", include=Phase.TEST)

        n.add(d1)
        # n.add(d2)
        n.add(l)
        n.add(l3)
        n.add(l4)

        return n.str().__str__()
