


from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
from sklearn.base import TransformerMixin
import numpy as np
import operator
from utils import write_pred

from sklearn.metrics import mean_squared_error

class EnsembleClassifier(BaseEstimator, ClassifierMixin, TransformerMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------
    clfs : array-like, shape = [n_classifiers]
      A list of scikit-learn classifier objects.

    weights : array-like, shape = [n_classifiers], optional (default=`None`)
      If `None`, the majority rule voting will be applied to the predicted
      class labels. If a list of weights (`float` or `int`) is provided,
      the averaged raw probabilities (via `predict_proba`) will be used to
      determine the most confident class label.

    Attributes
    ----------
    classes_ : array-like, shape = [n_class_labels, n_classifiers]
        Class labels predicted by each classifier if `weights=None`.

    probas_ : array, shape = [n_probabilities, n_classifiers]
        Predicted probabilities by each classifier if `weights=array-like`.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fits the scikit-learn estimators.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        y : array-like, shape = [n_samples]
            Target values.

        Returns
        -------
        self : object

        """
        for clf in self.clfs:
            print 'fitting:', clf.__class__.__name__
            clf.fit(X, y)
        return self

    def predict(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        maj : array-like, shape = [n_class_labels]
            Predicted class labels by majority rule.

        """
        # if self.weights:
        #     avg = self.predict_proba(X)
        #     maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)
        #
        # else:
        #     self.classes_ = self._get_classes(X)
        #     maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])
        #
        # return maj

        # if self.weights:
        #     return self.predict_proba(X)
        #     # import pdb
        #     # pdb.set_trace()

        preds = []
        for clf in self.clfs:
            print 'predicting:', clf.__class__.__name__
            preds.append(clf.predict(X))

        return np.asarray(preds)


    def predict_proba(self, X):

        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        ----------
        avg : array-like, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = self._get_probas(X)
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg


    def transform(self, X):
        """
        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            Training vectors, where n_samples is the number of samples and
            n_features is the number of features.

        Returns
        -------
        If not `weights=None`:
          array-like = [n_classifier_results, n_class_proba, n_class]
            Class probabilties calculated by each classifier.

        Else:
          array-like = [n_classifier_results, n_class_label]
            Class labels predicted by each classifier.

        """
        if self.weights:
            return self._get_probas(X)
        else:
            return self._get_classes(X)

    def _get_classes(self, X):
        """ Collects results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs])

    def _get_probas(self, X):
        """ Collects results from clf.predict calls. """
        return np.asarray([clf.predict(X) for clf in self.clfs])

def log(file, line):
    file.write(line + "\n")
    file.flush()
    print(line)

if __name__ == '__main__':
    from data import DataSet
    from sklearn.ensemble import *
    from sklearn.linear_model import *
    from sklearn.svm import SVR
    from sklearn import cross_validation
    from sklearn.cross_validation import KFold


    # num_of_models = 5


    clf2 = RandomForestRegressor()
    clf3 = ExtraTreesRegressor()
    clf4 = SVR()
    clf5 = GradientBoostingRegressor(n_estimators=300, learning_rate=0.1, loss='ls')
    clf6 = BaggingRegressor(n_estimators=10)
    clf7 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf8 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf9 = BaggingRegressor(base_estimator=clf2, n_estimators=10, max_samples=0.8)
    clf10 = AdaBoostRegressor(base_estimator=clf5)
    clf11 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf12 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)
    clf13 = BaggingRegressor(base_estimator=clf5, n_estimators=10, max_samples=0.8)

    # clf9 = GradientBoostingRegressor(base_estimator=clf6)


    # averager = LogisticRegression(fit_intercept=False)
    averager = LinearRegression(fit_intercept=False)
    clfs = [clf2, clf3, clf4, clf5, clf6, clf7, clf8, clf9, clf10, clf11, clf12, clf13]

    eclf = EnsembleClassifier(clfs=[clf2, clf6, clf5])
    data = DataSet('../dataset/train.npy')
    idx = data.idx_by_gradient_boost(topk=100)
    X = data.features_no_ingre[:,idx]
    # X = data.features_no_ingre
    y = data.labels
    logfout = open('log.txt','wb')

    kf = KFold(n=X.shape[0], n_folds=5)


    # round 1 for learning the weights for the classifiers

    errors = {}

    for clf in clfs:
        errors[clf.__class__.__name__] = []

    errors['mean_model'] = []

    for train, test in kf:
        y_train_preds = []
        y_test_preds = []
        for clf in clfs:
            log(logfout, 'fitting: ' + clf.__class__.__name__)
            clf.fit(X[train], y[train])
            y_train_pred = clf.predict(X[train])
            log(logfout, 'train score: %.4f'%mean_squared_error(y_train_pred, y[train]))
            y_test_pred = clf.predict(X[test])
            mse = mean_squared_error(y_test_pred, y[test])
            log(logfout, 'test score: %.4f'%mse)
            y_train_preds.append(y_train_pred)
            y_test_preds.append(y_test_pred)

            errors[clf.__class__.__name__].append(mse)

        # import pdb
        # pdb.set_trace()
        y_trains = np.asarray(y_train_preds)
        y_tests = np.asarray(y_test_preds)
        # import pdb
        # pdb.set_trace()
        # averager.fit(y_trains.swapaxes(0, 1), y[train])
        # y_test_pred = averager.predict(y_tests.swapaxes(0,1))
        # log(logfout, '..average model: ' + averager.__class__.__name__)
        y_test_pred = y_tests.swapaxes(0,1).mean(axis=1)
        mse = mean_squared_error(y_test_pred, y[test])
        log(logfout, 'mean test score: %.4f'%mse)
        errors['mean_model'].append(mse)
        print

    for k in errors:
        log(logfout, "%s: %.4f"%(k, np.mean(errors[k])))

    test = DataSet('../dataset/test.npy')
    test_X = test.features_no_ingre[:,idx]
    # test_X = test.features_no_ingre

    y_preds = []
    for clf in clfs:
        print clf.__class__.__name__
        clf.fit(X, y)
        y_preds.append(clf.predict(test_X))

    y_npy = np.asarray(y_preds).mean(axis=0)


    write_pred(filename='mean_model.csv', ids=test.ids, preds=y_npy)




    # import pdb
    # pdb.set_trace()

    # for clf in [clf2, clf4, clf6, clf5, eclf]:
    #
    #     scores = cross_validation.cross_val_score(clf, X, y, cv=5, scoring='mean_squared_error')
    #     print("Accuracy: %0.2f (+/- %0.2f) [%s]" % (scores.mean(), scores.std(), clf.__class__.__name__))
