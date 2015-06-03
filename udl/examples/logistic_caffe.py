from udl.caffei.models.logistic import Logistic
import sklearn
import sklearn.datasets
import sklearn.linear_model
from sklearn.metrics import accuracy_score


def get_data():
    X, Y = sklearn.datasets.make_classification(
        n_samples=10000, n_features=4, n_redundant=0, n_informative=2,
        n_clusters_per_class=2, hypercube=False, random_state=0
    )

    # Split into train and test
    X, Xt, Y, Yt = sklearn.cross_validation.train_test_split(X, Y)
    return X, Xt, Y, Yt


clf_caffe = Logistic()

X, Xt, Y, Yt = get_data()
# Train and test the scikit-learn SGD logistic regression.
clf = sklearn.linear_model.SGDClassifier(loss='log', n_iter=1000, penalty='l2', alpha=1e-3, class_weight='auto')
clf.fit(X, Y)
yt_pred = clf.predict(Xt)
print yt_pred
print('Accuracy: {:.3f}'.format(sklearn.metrics.accuracy_score(Yt, yt_pred)))

# Train and test the caffe logistic regression
clf = Logistic()
clf.fit(X, Y)
pred = clf.predict(Xt)
accuracy = accuracy_score(Yt, pred)

print('Accuracy from sklearn: {:.3f}'.format(sklearn.metrics.accuracy_score(Yt, yt_pred)))
print("Accuracy from caffe: {:.3f}".format(accuracy))
