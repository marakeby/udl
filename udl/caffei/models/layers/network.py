__author__ = 'haitham'
import udl.caffei.models.caffe_pb2 as pb2


class Network(object):
    def __init__(self):
        self._inner = pb2.NetParameter()

    def add(self, layer):
        self._inner.layer.add().CopyFrom(layer._layer)

    def str(self):
        return self._inner
