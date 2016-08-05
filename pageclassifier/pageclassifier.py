import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier


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
        self._clf = RandomForestClassifier(
                        class_weight='balanced').fit(X, labels)


    @_ensure_trained
    def predict(self, wikicode_list):
        features = [ext.extract(wikicode_list) for ext in self._fx_exts]
        X = np.concatenate(features, axis=1)
        return self._clf.predict(X)
