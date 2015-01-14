

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import numpy as np
# import pdb
# pdb.set_trace()

dir = '/Volumes/Storage/Unilever_Challenge/dataset'
with open(dir + '/train.npy') as train_in, open(dir +'/test.npy') as test_in:
	train = np.load(train_in)
	test = np.load(test_in)

train_labels = train[:, -1]
train_data = train[:, 158:-2]
# train_data = train[:10000, 2:158]
test_labels = test[:, -1]
# test_data = test[:10000, 158:-2]
test_data = test[:, 158:-2]

proc = StandardScaler()

X, y = train_data, train_labels
X = proc.fit_transform(X)

np.random.seed(123)

clf1 = LogisticRegression()
clf2 = RandomForestRegressor()
# clf3 = GaussianNB()
clf4 = BaggingRegressor(base_estimator=clf2, n_estimators=10)
clf5 = BaggingRegressor(base_estimator=clf4, n_estimators=10)

print('5-fold cross validation:\n')

for clf in [clf1, clf2, clf4, clf5]:

    scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='mean_squared_error')
#     y_pred = clf.predict(test_data)
#     scores = mean_squared_error(y_pred, test_labels)
    
    print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), clf.__class__.__name__))