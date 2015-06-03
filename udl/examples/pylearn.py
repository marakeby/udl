__author__ = 'haitham'
from udl.pylearni.models.Autoencoder import Autoencoder
from udl.caffei.models.old.logistic import Logistic
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score

from data import datagen

X, Xt, Y, Yt = datagen.get_sklearn_data()

auto = Autoencoder(4)
clf_caffe = Logistic()

pipe = Pipeline([('autoencoder_pylearn2', auto), ('logistic_caffe', clf_caffe)])
pipe.fit(X, Y)
pred = pipe.predict(Xt)

accuracy = accuracy_score(Yt, pred)
print("Accuracy from caffe: {:.3f}".format(accuracy))
