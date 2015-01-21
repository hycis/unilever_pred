

from sklearn import cross_validation
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB 
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.ensemble import BaggingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import numpy as np
from utils import write_pred
# import pdb
# pdb.set_trace()

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt



dir = '/Volumes/Storage/Unilever_Challenge/dataset'
with open(dir + '/train.npy') as train_in, open(dir +'/test.npy') as test_in:
	train = np.load(train_in)
	test = np.load(test_in)

train_labels = train[:, -1]
train_data = train[:, 158:-2]
# train_data = train[:10000, 2:158]

test_labels = test[:, -1]
test_data = test[:, 1:-2]
# test_data = test[:10000, 2:158]

# import pdb
# pdb.set_trace()
# proc = StandardScaler()

X, y = train_data, train_labels
# X = proc.fit_transform(X)

np.random.seed(123)

clf1 = LogisticRegression()
clf2 = RandomForestRegressor()
clf3 = RandomForestClassifier()
clf4 = BaggingRegressor(n_estimators=10)
clf6 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, max_depth=2, random_state=0, loss='ls')
clf5 = BaggingRegressor(base_estimator=clf6, n_estimators=10)

n_folds = 5

for clf in [clf1, clf3]:
	
	scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='accuracy', verbose=True)
	# import pdb
# 	pdb.set_trace()	
	print("Accuracy: %0.5f (+/- %0.5f) [%s]" % (scores.mean(), scores.std(), clf.__class__.__name__))

	
# 	import pdb
# 	pdb.set_trace()
	
# 	clf.fit(X,y)
# 	import pdb
# 	pdb.set_trace()
# 	print 'saving'
# 	write_pred(filename='%s.csv'%clf.__class__.__name__, ids=map(lambda x: int(round(x)), 
# 			   test[:, 0]), preds=clf.predict(test_data))
	
	
# 	fig, ax = plt.subplots()
# # 	import pdb
# # 	pdb.set_trace()
# 	data = clf.feature_importances_
# 	rects1 = ax.bar([a for a in range(1, 1+data.shape[0])], data)
# 	plt.show()
	


