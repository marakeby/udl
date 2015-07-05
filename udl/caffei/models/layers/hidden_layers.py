import inspect

__author__ = 'haitham'
import udl.caffei.models.caffe_pb2 as pb2
from udl.caffei.models.layers.layer import Layer, set

def get_cong(frame):
    args, _, _, values = inspect.getargvalues(frame)
    args_to_remove = ['self', 'frame', 'name', 'top', 'bottom', 'include', 'exclude']

    conf = {key: value for key, value in values.items()
         if (key not in args_to_remove) and (value is not None)}
    return conf

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

        # if weight_filler is None:
        #     self.inner_product_param.weight_filler.CopyFrom(Filler().str())
        # else:
        #     self.weight_filler = weight_filler
        #
        # if bias_filler is None:
        #     self.inner_product_param.bias_filler.CopyFrom(Filler().str())
        # else:
        #     self.bias_filler = bias_filler

        self._layer.inner_product_param.CopyFrom(self.inner_product_param)
        # self.bias_filler = bias_filler
        self.bias_term = bias_term
        if bias_filler is not None:
            self.bias_filler = bias_filler
        if weight_filler is not None:
            self.weight_filler = weight_filler
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
        # set(self.inner_product_param.weight_filler, val)
        self._layer.inner_product_param.weight_filler.CopyFrom(val)
    @property
    def bias_filler(self):
        return self.inner_product_param.bias_filler

    @bias_filler.setter
    def bias_filler(self, val):
        # set(self.inner_product_param.bias_filler, val)
        self._layer.inner_product_param.bias_filler.CopyFrom(val)

    def str(self):
        return self._layer



class Convolution(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                num_output=1,
                bias_term=None,
                pad=None,
                pad_h=None,
                pad_w=None,
                kernel_size=None,
                kernel_h=None,
                kernel_w=None,
                group=None,
                stride=None,
                stride_h=None,
                stride_w=None,
                weight_filler = None,
                bias_filler= None,
                engine= None):
                # engine= pb2.ConvolutionParameter.DEFAULT):
                # engine= None):
        super(Convolution, self).__init__(name, top, bottom, "Convolution", None, include, exclude)

        # conf ={
        # 'num_output' : num_output,
        # 'bias_term' : bias_term,
        # 'pad' : pad,
        # #'pad_h' : pad_h,
        # #'pad_w' : pad_w,
        # 'kernel_size' : kernel_size,
        # #'kernel_h' : kernel_h,
        # #'kernel_w' : kernel_w,
        # 'group' : group,
        # 'stride' : stride,
        # #'stride_h' : stride_h,
        # #'stride_w' : stride_w,
        # # 'weight_filler' : weight_filler,
        # # 'bias_filler' : bias_filler,
        # 'engine' : engine}
        # if weight_filler is None:
        #     weight_filler = Filler().str()
        # else:
        #     weight_filler= weight_filler .str()
        #
        # if bias_filler is None:
        #     bias_filler = Filler()
        # else:
        #     bias_filler= bias_filler .str()

        frame = inspect.currentframe()
        conf = get_cong(frame)

        convolution_param = pb2.ConvolutionParameter(**conf)



        # convolution_param.weight_filler.CopyFrom(weight_filler.str())
        #
        # convolution_param.bias_filler.CopyFrom(bias_filler.str())


        self._layer.convolution_param.CopyFrom(convolution_param)

    def str(self):
        return self._layer

class Pooling(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                pool=pb2.PoolingParameter.DEFAULT,
                # bias_term=True,
                pad=None,
                pad_h=None,
                pad_w=None,
                kernel_size=None,
                kernel_h=None,
                kernel_w=None,
                stride=None,
                stride_h=None,
                stride_w=None,
                engine= None,
                global_pooling = None):
        super(Pooling, self).__init__(name, top, bottom, "Pooling", None, include, exclude)

        frame = inspect.currentframe()
        conf = get_cong(frame)

        # conf ={
        # 'pool' : pool,
        # 'pad' : pad,
        # #'pad_h' : pad_h,
        # #'pad_w' : pad_w,
        # 'kernel_size' : kernel_size,
        # #'kernel_h' : kernel_h,
        # #'kernel_w' : kernel_w,
        # 'stride' : stride,
        # #'stride_h' : stride_h,
        # #'stride_w' : stride_w,
        # 'global_pooling' : global_pooling,
        # 'engine' : engine}
        # print conf
        convolution_param = pb2.PoolingParameter(**conf)
        self._layer.pooling_param.CopyFrom(convolution_param)

    def str(self):
        return self._layer


class Relu(Layer):
    def __init__(self, name="name", top=None, bottom=None, include=None, exclude=None,
                negative_slope=None,
                # engine= pb2.PoolingParameter.DEFAULT,
                engine= None,
                ):
        super(Relu, self).__init__(name, top, bottom, "ReLU", None, include, exclude)

        frame = inspect.currentframe()
        conf = get_cong(frame)
        # conf ={
        # 'negative_slope': negative_slope,
        # 'engine' : engine}


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
        super(LRN, self).__init__(name, top, bottom, "LRN", None, include, exclude)

        frame = inspect.currentframe()
        conf = get_cong(frame)

        # conf ={
        # 'local_size' : local_size,
        # 'alpha' : alpha,
        # 'beta' : beta,
        # 'norm_region' : norm_region,
        # 'k' : k
        # }


        lrn_param = pb2.LRNParameter(**conf)
        self._layer.lrn_param.CopyFrom(lrn_param)

    def str(self):
        return self._layer
