__author__ = 'haitham'
from udl.caffei.models.layers.layer import Layer, set
import udl.caffei.models.caffe_pb2 as pb2


class DummyDataLayer(Layer):
    def __init__(self, name="name", top=None, include=None, exclude=None, params=None):
        super(DummyDataLayer, self).__init__(name, top, None, "DummyData", None, include, exclude)
        self.params = params
        self._layer.dummy_data_param.CopyFrom(params.str())

    def str(self):
        return self._layer


class DummyDataParameter(object):
    def __init__(self, num, channels, height, width):
        self.params = pb2.DummyDataParameter()
        self.num = num
        self.channels = channels
        self.height = height
        self.width = width

    @property
    def num(self):
        return self.params.num

    @num.setter
    def num(self, val):
        set(self.params.num, val)

    @property
    def channels(self):
        return self.params.channels

    @channels.setter
    def channels(self, val):
        set(self.params.channels, val)

    @property
    def height(self):
        return self.params.height

    @height.setter
    def height(self, val):
        set(self.params.height, val)

    @property
    def width(self):
        return self.params.width

    @width.setter
    def width(self, val):
        set(self.params.width, val)

    def str(self):
        return self.params

# EltwiseLayer< Dtype >
# FlattenLayer< Dtype >
# HDF5DataLayer< Dtype >
# HDF5OutputLayer< Dtype >
# Im2colLayer< Dtype >

# loss

# caffe::LossLayer< Dtype >
# caffe::ContrastiveLossLayer< Dtype >
# caffe::EuclideanLossLayer< Dtype >
# caffe::HingeLossLayer< Dtype >
# caffe::InfogainLossLayer< Dtype >
# caffe::MultinomialLogisticLossLayer< Dtype >
# caffe::SigmoidCrossEntropyLossLayer< Dtype >
# caffe::SoftmaxWithLossLayer< Dtype >

# Neuron

# common
# inner product
