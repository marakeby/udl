__author__ = 'haitham'

from udl.caffei.models.layers.layer import Layer, set
import udl.caffei.models.caffe_pb2 as pb2


class LossType():
    HINGE_LOSS = "HINGE_LOSS"
    CONTRASTIVE_LOSS = "CONTRASTIVE_LOSS"
    EUCLIDEAN_LOSS = "EUCLIDEAN_LOSS"
    MULTINOMIAL_LOGISTIC_LOSS = "MULTINOMIAL_LOGISTIC_LOSS"
    SoftmaxWithLoss = "SoftmaxWithLoss"
    SOFTMAX = "Softmax"
    SIGMOID_CROSS_ENTROPY_LOSS = "SIGMOID_CROSS_ENTROPY_LOSS"


class Loss(Layer):
    def __init__(self, name, type=None, top=None, bottom=None, include=None, exclude=None, params=None):
        super(Loss, self).__init__(name, top, bottom, type, None, include, exclude)
        if type == LossType.HINGE_LOSS:
            self._layer.hinge_loss_param.CopyFrom(params.str())
        self.params = params

    def str(self):
        return self._layer


class HingeLossParamters(object):
    L1 = pb2.HingeLossParameter.L1
    L2 = pb2.HingeLossParameter.L2

    def __init__(self, norm=L2):
        print norm
        self.params = pb2.HingeLossParameter()
        self.norm = norm
        print self.params.norm

    @property
    def norm(self):
        return self.params.norm

    @norm.setter
    def norm(self, val):
        self.params.norm = set(self.params.norm, val)

    def str(self):
        return self.params


class ContrastiveLossParameter(object):
    def __init__(self, margin):
        self.params = pb2.ContrastiveLossParameter()
        self.margin = margin

    @property
    def margin(self):
        return self.params.margin

    @margin.setter
    def margin(self, val):
        self.params.margin = set(self.params.margin, val)

    def str(self):
        return self.params


class ContrastiveLossParameter(object):
    def __init__(self, margin):
        self.params = pb2.ContrastiveLossParameter()
        self.margin = margin

    @property
    def margin(self):
        return self.params.margin

    @margin.setter
    def margin(self, val):
        self.params.margin = set(self.params.margin, val)

    def str(self):
        return self.params
