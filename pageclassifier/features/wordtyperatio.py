import csv
from . import utils
import numpy as np


class FeatureExtractor(object):

    def __init__(self, wordtype_filepaths):
        self._wordtypes = []
        for fp in wordtype_filepaths:
            with open(fp) as f:
                reader = csv.reader(f)
                self._wordtypes.append(set([r[0] for r in reader]))

    def fit_extract(self, wikicode_list):
        return self.extract(wikicode_list)

    def extract(self, wikicode_list):
        X = []
        for wcode in wikicode_list:
            X.append(self._extract_for_single_wikicode(wcode))
        return np.array(X)

    def _extract_for_single_wikicode(self, wikicode):
        result = []
        words = utils.select_words_from_wikicode(wikicode)
        wordset = set([w.lower() for w in words])
        for typeset in self._wordtypes:
            result.append(len(wordset.difference(typeset))/len(wordset))
        return result
