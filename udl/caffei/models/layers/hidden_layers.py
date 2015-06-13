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



class Convolution(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                num_output=1,
                bias_term=True,
                pad=0,
                pad_h=0,
                pad_w=0,
                kernel_size=0,
                kernel_h=0,
                kernel_w=0,
                group=1,
                stride=0,
                stride_h=0,
                stride_w=0,
                weight_filler = None,
                bias_filler= None,
                engine= pb2.ConvolutionParameter.DEFAULT):
        super(Convolution, self).__init__(name, top, bottom, "Convolution", None, include, exclude)

        conf ={
        'num_output' : num_output,
        'bias_term' : bias_term,
        'pad' : pad,
        'pad_h' : pad_h,
        'pad_w' : pad_w,
        'kernel_size' : kernel_size,
        'kernel_h' : kernel_h,
        'kernel_w' : kernel_w,
        'group' : group,
        'stride' : stride,
        'stride_h' : stride_h,
        'stride_w' : stride_w,
        # 'weight_filler' : weight_filler,
        # 'bias_filler' : bias_filler,
        'engine' : engine}


        convolution_param = pb2.ConvolutionParameter(**conf)


        if weight_filler is None:
            convolution_param.weight_filler.CopyFrom(Filler().str())


        if bias_filler is None:
            convolution_param.bias_filler.CopyFrom(Filler().str())


        self._layer.convolution_param.CopyFrom(convolution_param)

    def str(self):
        return self._layer

class Pooling(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                pool=pb2.PoolingParameter.DEFAULT,
                # bias_term=True,
                pad=0,
                pad_h=0,
                pad_w=0,
                kernel_size=0,
                kernel_h=0,
                kernel_w=0,
                stride=0,
                stride_h=0,
                stride_w=0,
                engine= pb2.PoolingParameter.DEFAULT,
                global_pooling = False):
        super(Pooling, self).__init__(name, top, bottom, "Pooling", None, include, exclude)

        conf ={
        'pool' : pool,
        'pad' : pad,
        'pad_h' : pad_h,
        'pad_w' : pad_w,
        'kernel_size' : kernel_size,
        'kernel_h' : kernel_h,
        'kernel_w' : kernel_w,
        'stride' : stride,
        'stride_h' : stride_h,
        'stride_w' : stride_w,
        'global_pooling' : global_pooling,
        'engine' : engine}

        convolution_param = pb2.PoolingParameter(**conf)
        self._layer.pooling_param.CopyFrom(convolution_param)

    def str(self):
        return self._layer


class Relu(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                negative_slope=0,
                engine= pb2.PoolingParameter.DEFAULT,
                ):
        super(Relu, self).__init__(name, top, bottom, "Relu", None, include, exclude)

        conf ={
        'negative_slope': negative_slope,
        'engine' : engine}


        relu_param = pb2.ReLUParameter(**conf)

        self._layer.relu_param.CopyFrom(relu_param)

    def str(self):
        return self._layer

class LRN(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                local_size=5,

                alpha=1,
                beta=0.75,
                norm_region=0,
                k=0,
               ):
        super(LRN, self).__init__(name, top, bottom, "Pooling", None, include, exclude)

        conf ={
        'local_size' : local_size,
        'alpha' : alpha,
        'beta' : beta,
        'norm_region' : norm_region,
        'k' : k
        }


        lrn_param = pb2.LRNParameter(**conf)
        self._layer.lrn_param.CopyFrom(lrn_param)

    def str(self):
        return self._layer
