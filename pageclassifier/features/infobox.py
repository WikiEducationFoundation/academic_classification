from sklearn.feature_extraction.text import TfidfVectorizer


def _filter_infobox_names(wcode):
    tmplts = wcode.filter_templates()
    return [str(t.name).strip() for t in tmplts
            if 'infobox' in str(t.name).lower()]


def _transform(wcode_list):
    result = []
    for wcode in wcode_list:
        infobox_names = _filter_infobox_names(wcode)
        cleaned = [n.lower().replace(' ', '-') for n in infobox_names]
        result.append(' '.join(cleaned))
    return result


class FeatureExtractor(object):

    def __init__(self):
        self._vectorizer = None

    def fit_extract(self, wikicode_list):
        self._vectorizer = TfidfVectorizer(min_df=0.02)
        transformed = _transform(wikicode_list)
        X = self._vectorizer.fit_transform(transformed)
        return X.toarray()

    def extract(self, wikicode_list):
        transformed = _transform(wikicode_list)
        X = self._vectorizer.transform(transformed)
        return X.toarray()
