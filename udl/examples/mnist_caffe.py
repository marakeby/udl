import sklearn

from udl.examples.data import datagen
from caffei.models.lenet import LeNet

x_train,y_train, x_test,y_test = datagen.get_mnist_data('/home/haitham/data/mnist')
x_train = x_train * 0.00390625 # normalization
x_test = x_test* 0.00390625

n_samples= x_train.shape[0]
x_train = x_train.reshape((n_samples,1,28,28))
n_samples= x_test.shape[0]
x_test = x_test.reshape((n_samples,1,28,28))

print x_train.shape,
print y_train.shape,
print x_test.shape,
print y_test.shape

# clf = sklearn.linear_model.SGDClassifier(loss='log', n_iter=100, penalty='l2', alpha=1e-3, class_weight='auto')
clf =LeNet()
clf.fit(x_train, y_train)
yt_pred = clf.predict(x_test)
print yt_pred.shape
print('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(y_test, yt_pred)))




