import caffe
import numpy as np
from udl.caffei.datasets.DummyDataset import DummyDataset
from udl.caffei.models.layers.innerproduct_pb2 import InnerProduct
from udl.caffei.models.layers.layer import Phase
from udl.caffei.models.layers.loss_layers import Loss, LossType
from udl.caffei.models.layers.network import Network
from udl.caffei.models.utils import get4D
from udl.caffei.trainers.SGDTrainer import SGDTrainer
from udl.model import UDLModel

class Logistic(UDLModel):
    def __init__(self,  dataset_adaptor=DummyDataset(), trainer=SGDTrainer()):
        caffe.set_mode_cpu()
        self.dataset_adaptor = dataset_adaptor
        self.trainer = trainer


    def fit(self, X, Y):
        X4D = get4D(X)
        Y4D = get4D(Y)
        data_layer = self.dataset_adaptor.fit_transform(X4D,Y4D)
        net = self.get_net(data_layer)
        self.trainer.init_trainer(net,X4D, Y4D)
        self.trainer.solve()
        return self.trainer


    def predict(self, x_test):
        self.trainer.predict (x_test)
        prob = self.trainer.solver.test_nets[0].blobs['output'].data
        print prob.shape
        pred = np.argmax(prob, axis=1)
        return pred

    def get_net(self, data_layer):
        n = Network()
        n.add(data_layer)
        n.name = "LogisticRegressionNet"
        l = InnerProduct(name="fc1", top="fc1", bottom="data", num_output=2)
        l3 = Loss(name="loss", type=LossType.SoftmaxWithLoss, bottom=["fc1", "label"], include=Phase.TRAIN)
        l4 = Loss(name="output", type= LossType.SOFTMAX, top= "output", bottom="fc1", include=Phase.TEST)

        n.add(l)
        n.add(l3)
        n.add(l4)

        return n.str().__str__()
