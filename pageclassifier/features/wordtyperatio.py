import csv
from . import utils
import numpy as np
from sklearn.preprocessing import StandardScaler


class FeatureExtractor(object):

    def __init__(self, wordtype_filepaths):
        self._wordtypes = []
        self._scaler = StandardScaler()
        for fp in wordtype_filepaths:
            with open(fp) as f:
                reader = csv.reader(f)
                self._wordtypes.append(set([r[0] for r in reader]))

    def fit_extract(self, wikicode_list):
        return self._scaler.fit_transform(
            self._basic_transform(wikicode_list)
        )

    def extract(self, wikicode_list):
        return self._scaler.transform(
            self._basic_transform(wikicode_list)
        )

    def _basic_transform(self, wikicode_list):
        X = []
        for wcode in wikicode_list:
            X.append(self._extract_for_single_wikicode(wcode))
        return np.array(X)

    def _extract_for_single_wikicode(self, wikicode):
        result = []
        words = utils.select_words_from_wikicode(wikicode)
        wordset = set([w.lower() for w in words])
        for typeset in self._wordtypes:
            num_words = max(1, len(wordset))
            result.append(len(wordset.difference(typeset))/num_words)
        return result
