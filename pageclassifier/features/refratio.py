import refclassifier as rc
import csv
from wikirefs.bibliography import Bibliography


def pull_refs(wcode):
    b = Bibliography(wcode)
    return b.refs()


class FeatureExtractor(object):

    def __init__(self, training_file):
        with open(training_file) as f:
            reader = csv.reader(f)
            rows = [(r[1], (r[2] == "True")) for r in reader]
            refs, labels = zip(*rows)
        self._clf = rc.RefClassifier(rc.transform_ref_text)
        self._clf.train(refs, labels)

    def fit_extract(self, wikicode_list):
        return self.extract(page_text_list)

    def extract(self, wikicode_list):
        return [self._calc_ref_ratio(wcode) for wcode in wikicode_list]

    def _calc_ref_ratio(self, wcode):
        refs_counts = pull_refs(wcode)
        refs = [r['ref'] for r in refs]
        count = [r['count'] for r in refs]
        predictions = self._clf.predict([r for r in refs])
        positives = sum(c*p for c, p in zip(count, predictions))
        return positives / sum(count)
