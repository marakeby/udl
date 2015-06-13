import types

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

class MemoryData(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                batch_size=0,
                channels=0,
                height=0,
                width=0,
                ):
        super(MemoryData, self).__init__(name, top, bottom, "MemoryData", None, include, exclude)

        conf ={
        'batch_size' : batch_size,
        'channels' : channels,
        'height' : height,
        'width' : width,
        }

        memory_data_param = pb2.MemoryDataParameter(**conf)
        self._layer.memory_data_param.CopyFrom(memory_data_param)

    def str(self):
        return self._layer

class Data(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                source='',
                batch_size=0,
                rand_skip=0,
                backend=0,
                scale=1,
                mean_file='',
                crop_size=0,
                mirror=False,
                force_encoded_color=False,

                ):
        super(Data, self).__init__(name, top, bottom, "Data", None, include, exclude)

        conf ={
        'source' : source,
        'batch_size' : batch_size,
        'rand_skip' : rand_skip,
        'backend' : backend,
        'scale' : scale,
        'mean_file' : mean_file,
        'crop_size' : crop_size,
        'mirror' : mirror,
        'force_encoded_color' : force_encoded_color,
        }

        data_param = pb2.DataParameter(**conf)
        self._layer.data_param.CopyFrom(data_param)

    # def __getattribute__(self, name):
    #     attr = super(Data, self).__getattribute__(name)
    #     if type(attr) == None:
    #         attr = self._layer.data_param.__getattribute__(name)
    #     return attr

    def str(self):
        return self._layer


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
