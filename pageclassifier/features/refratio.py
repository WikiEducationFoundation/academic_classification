import refclassifier as rc
import csv
from wikirefs.bibliography import Bibliography
import numpy as np
from sklearn.preprocessing import StandardScaler


def pull_refs(wcode):
    b = Bibliography(wcode)
    return b.refs()


class FeatureExtractor(object):

    def __init__(self, training_file):
        self._scaler = StandardScaler()
        with open(training_file) as f:
            reader = csv.reader(f)
            rows = [(r[1], (r[2] == "True")) for r in reader]
            refs, labels = zip(*rows)
        self._clf = rc.RefClassifier(rc.transform_ref_text)
        self._clf.train(refs, labels)

    def fit_extract(self, wikicode_list):
        return self._scaler.fit_transform(
            self._basic_transform(wikicode_list)
        )

    def extract(self, wikicode_list):
        return self._scaler.transform(
            self._basic_transform(wikicode_list)
        )

    def _basic_transform(self, wikicode_list):
        return np.array(
                    [[self._calc_ref_ratio(wcode)] for wcode in wikicode_list]
                    )

    def _calc_ref_ratio(self, wcode):
        refs_counts = pull_refs(wcode)
        refs = [r['ref'] for r in refs_counts]
        count = [r['count'] for r in refs_counts]
        total = sum(count)
        if total == 0:
            total = 1
        positives = 0
        if len(refs) > 0:
            predictions = self._clf.predict([r for r in refs])
            positives = sum([c*p for c, p in zip(count, predictions)])
        return positives / total
