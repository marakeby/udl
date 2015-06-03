__author__ = 'haitham'
import udl.caffei.models.caffe_pb2 as pb2
from udl.caffei.models.layers.layer import Layer, set


class Filler(object):
    def __init__(self, type="constant", value=0, min=0, max=1, mean=0, std=1, sparse=-1):
        self._filler = pb2.FillerParameter()
        self.type = type
        self.value = value
        self.min = min
        self.max = max
        self.mean = mean
        self.std = std
        self.sparse = sparse

    @property
    def type(self):
        return self._filler.type

    @type.setter
    def type(self, val):
        self._filler.type = set(self._filler.type, val)

    @property
    def value(self):
        return self._filler.value

    @value.setter
    def value(self, val):
        self._filler.value = set(self._filler.value, val)

    @property
    def min(self):
        return self._filler.min

    @min.setter
    def min(self, val):
        self._filler.min = set(self._filler.min, val)

    @property
    def max(self):
        return self._filler.max

    @max.setter
    def max(self, val):
        self._filler.max = set(self._filler.max, val)

    @property
    def mean(self):
        return self._filler.mean

    @mean.setter
    def mean(self, val):
        self._filler.mean = set(self._filler.mean, val)

    @property
    def std(self):
        return self._filler.std

    @std.setter
    def std(self, val):
        self._filler.std = set(self._filler.std, val)

    @property
    def sparse(self):
        return self._filler.sparse

    @sparse.setter
    def sparse(self, val):
        self._filler.sparse = set(self._filler.sparse, val)

    def str(self):
        return self._filler


class InnerProduct(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None, num_output=1, bias_term=None,
                 weight_filler=None, bias_filler=None):
        super(InnerProduct, self).__init__(name, top, bottom, "InnerProduct", None, include, exclude)


        # self._layer.inner_product_param.CopyFrom( params.str())
        self.inner_product_param = pb2.InnerProductParameter()
        # x = self._layer.inner_product_param
        self.num_output = num_output
        if weight_filler is None:
            self.inner_product_param.weight_filler.CopyFrom(Filler().str())
        else:
            self.weight_filler = weight_filler

        if bias_filler is None:
            self.inner_product_param.bias_filler.CopyFrom(Filler().str())
        else:
            self.bias_filler = bias_filler

        self._layer.inner_product_param.CopyFrom(self.inner_product_param)
        # self.bias_filler = bias_filler
        self.bias_term = bias_term
        # x = self._layer.inner_product_param.add()

        # x.weight_filler.std=0.01
        # self.layer.inner_product_param.weight_filler.type ="constant"

    @property
    def num_output(self):
        return self._layer.inner_product_param.num_output

    @num_output.setter
    def num_output(self, val):
        self.inner_product_param.num_output = set(self._layer.inner_product_param.num_output, val)

    @property
    def bias_term(self):
        return self._layer.inner_product_param.bias_term

    @bias_term.setter
    def bias_term(self, val):
        set(self.inner_product_param.bias_term, val)

    @property
    def weight_filler(self):
        return self.inner_product_param.weight_filler

    @weight_filler.setter
    def weight_filler(self, val):
        set(self.inner_product_param.weight_filler, val)

    @property
    def bias_filler(self):
        return self.inner_product_param.bias_filler

    @bias_filler.setter
    def bias_filler(self, val):
        set(self.inner_product_param.bias_filler, val)

    def str(self):
        return self._layer
