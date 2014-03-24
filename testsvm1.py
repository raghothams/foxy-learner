
from sklearn.datasets import load_svmlight_file
from sklearn import svm
import time
import numpy as np

t0 = time.time()
X_train, y_train = load_svmlight_file("data/tiny-train.csv", 
                                multilabel = True)
print "train load Done in ", time.time() - t0


clf = svm.SVC(kernel='linear')
t0 = time.time()
clf.fit(X_train, y_train)
print time.time()-t0

t0 = time.time()
X_test, y_test = load_svmlight_file("data/tiny-test.csv", 
                                multilabel = True, n_features=X_train.shape[1])
print "test load Done in ", time.time() - t0

t0 = time.time()
Z = clf.predict(X_test)
np.savetxt('res.txt', Z, delimiter=" ", fmt="%s")

print "predicted in ", time.time() - t0

