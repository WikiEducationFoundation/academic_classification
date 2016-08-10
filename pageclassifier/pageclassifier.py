import numpy as np
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.grid_search import GridSearchCV


def _ensure_trained(func):
    def wrapper(*args, **kwargs):
        if args[0]._clf is None:
            raise NotTrainedError
        return func(*args, **kwargs)
    return wrapper


class PageClassifier(object):

    def __init__(self, feature_extractors):
        self._fx_exts = feature_extractors
        self._clf = None

    def train(self, wikicode_list, labels):
        features = [ext.fit_extract(wikicode_list) for ext in self._fx_exts]
        X = np.concatenate(features, axis=1)
        kbest = SelectKBest(f_classif)
        # model = RandomForestClassifier(
        #             class_weight='balanced')
        # model = GradientBoostingClassifier()
        model = LinearSVC(class_weight='balanced',
                          dual=False,
                          penalty='l1')
        pipe = Pipeline([('kbest', kbest), ('model', model)])
        self._clf = GridSearchCV(pipe,
                                 {'kbest__k': list(range(1, X.shape[1], 10))},
                                 scoring='roc_auc',
                                 cv=10
                                 ).fit(X, labels)

    @_ensure_trained
    def predict(self, wikicode_list):
        X = self._extract_feature_vectors_from_wikicode_list(wikicode_list)
        return self._clf.predict(X)

    @_ensure_trained
    def predict_proba(self, wikicode_list):
        X = self._extract_feature_vectors_from_wikicode_list(wikicode_list)
        # return [cls1 for cls0, cls1 in self._clf.predict_proba(X)]
        return self._clf.decision_function(X)


    def _extract_feature_vectors_from_wikicode_list(self, wikicode_list):
        features = [ext.extract(wikicode_list) for ext in self._fx_exts]
        X = np.concatenate(features, axis=1)
        return X
