from sklearn.feature_extraction.text import TfidfVectorizer
from . import utils


def _transform(n, wikicode_list):
    result = []
    for wcode in wikicode_list:
        words = utils.select_words_from_wikicode(wcode, first_n=n)
        joined = ' '.join(words)
        result.append(joined)
    return result


class FeatureExtractor(object):

    def __init__(self, n):
        self._n = n
        self._vectorizer = None

    def fit_extract(self, wikicode_list):
        first_n_words = _transform(self._n, wikicode_list)
        self._vectorizer = TfidfVectorizer(min_df=0.01,
                                           max_df=0.15,
                                           stop_words='english')
        X = self._vectorizer.fit_transform(first_n_words)
        return X.toarray()

    def extract(self, wikicode_list):
        first_n_words = _transform(self._n, wikicode_list)
        X = self._vectorizer.transform(first_n_words)
        return X.toarray()
