from google.protobuf.internal.containers import RepeatedScalarFieldContainer, RepeatedCompositeFieldContainer

__author__ = 'haitham'
import udl.caffei.models.caffe_pb2 as pb2


class Layer(object):
    def __init__(self, name="name", top=None, bottom=None, layer_type=None, loss_weight=None, include=None,
                 exclude=None):
        self._layer = pb2.LayerParameter()
        self._layer.name = name

        
        self.layer_type = layer_type
        if top == None:
            top = name
        self.top = top

        self.bottom = bottom

        self.include = include

        self.exclude = exclude
        self.loss_weight = loss_weight

    @property
    def bottom(self):
        return self._layer.bottom

    @bottom.setter
    def bottom(self, val):
        set(self._layer.bottom, val)

    @property
    def top(self):
        return self._layer.top

    @top.setter
    def top(self, val):
        set(self._layer.top, val)

    @property
    def layer_type(self):
        return self._layer.type

    @layer_type.setter
    def layer_type(self, val):
        self._layer.type = set(self._layer.type, val)

    @property
    def loss_weight(self):
        return self._layer.loss_weight

    @loss_weight.setter
    def loss_weight(self, val):
        set(self._layer.loss_weight, val)

    @property
    def include(self):
        return self._layer.include

    @include.setter
    def include(self, val):
        if val is None:
            return
        x = self._layer.include.add()
        x.phase = val
        # set(self._layer.include, val)

    @property
    def exclude(self):
        return self._layer.exclude

    @exclude.setter
    def exclude(self, val):
        if val is None:
            return
        x = self._layer.exclude.add()
        x.phase = val


class Phase():
    TRAIN = pb2.TRAIN
    TEST = pb2.TEST


def set(prop, val):
    if val is None:
        return
    if type(prop) is RepeatedScalarFieldContainer:
        if type(val) is list:
            for i in val:
                prop.append(i)
        else:
            prop.append(val)
    elif type(prop) is RepeatedCompositeFieldContainer:
        x = prop.add()
        # x.phase= val
        x = val
    else:
        prop = val
    return prop  # use for immutable fields
