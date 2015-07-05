import numpy as np
import caffe
from udl.caffei.datasets.DummyDataset_batch import DummyDataset_batch
from udl.caffei.models.layers.layer import Phase
from udl.caffei.models.layers.loss_layers import Loss, LossType
from udl.caffei.models.layers.network import Network
from udl.caffei.models.utils import get4D
from udl.caffei.trainers.SGDTrainer import SGDTrainer
from udl.model import UDLModel
import udl.caffei.models.caffe_pb2 as pb2
from udl.caffei.models.layers.hidden_layers import Convolution, Pooling, Relu, InnerProduct


class LeNet(UDLModel):
    def __init__(self,  dataset_adaptor=DummyDataset_batch(), trainer=SGDTrainer()):
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
        net = Network()

        net.add(data_layer)
        net.name = "LeNet"
        c_bottom = ['data', 'pool0']
        c_num_output = [20, 50]

        max= pb2.PoolingParameter.MAX
        xavier = pb2.FillerParameter(type= 'xavier')
        const =  pb2.FillerParameter(type= 'constant')
        for i in range(0,2):
            c = 'conv%d'%i
            p = "pool%d"%i
            r = "relu%d"%i
            n = "norm%d"%i
            c_layer = Convolution(name=c , bottom= c_bottom[i] , num_output = c_num_output[i] , kernel_size =5, stride=1, weight_filler= xavier , bias_filler = const)
            p_layer = Pooling(name= p, bottom= c, kernel_size=2 , stride =2, pool = max )
            net.add(c_layer), net.add(p_layer),

        ip1 = InnerProduct(name="ip1", bottom="pool1", num_output=500, weight_filler= xavier , bias_filler = const)
        r_layer = Relu(name = r, bottom= "ip1",  top= "ip1")
        ip2 = InnerProduct(name="ip2", bottom="ip1", num_output=10,  weight_filler= xavier , bias_filler = const)

        o1 = Loss(name="loss", type=LossType.SoftmaxWithLoss, bottom=["ip2", "label"], include=Phase.TRAIN)
        o2 = Loss(name="output", type= LossType.SOFTMAX, top= "output", bottom="ip2", include=Phase.TEST)

        net.add(ip1), net.add(r_layer), net.add(ip2), net.add(o1), net.add(o2)


        return net.str().__str__()
