from udl.caffei.models.layers.data_layers import DummyDataLayer, DummyDataParameter

__author__ = 'haitham'
class DummyDataset():
    def fit_transform(self, X, y=None, **fit_params):

        xsh = X.shape
        ysh =y.shape
        data_layer=  DummyDataLayer(name='data', top=['data', 'label'],
                            params=DummyDataParameter([xsh[0], ysh[0]], [xsh[1], ysh[1]], [xsh[2], ysh[2]], [xsh[3], ysh[3]]))

        return data_layer