from sklearn.feature_extraction.text import TfidfVectorizer
import re
import mwparserfromhell.nodes as mw_nodes
from mwparserfromhell.wikicode import Wikicode


def _select_first_n_words_from_wikicode(n, wikicode):
    first_words = []
    for node in wikicode.nodes:
        text = _extract_text_from_node(node)
        words = _extract_words_from_text(text)
        first_words.extend(words)
        if len(first_words) > n:
            break
    return first_words[:n]


def _extract_words_from_text(text):
    return [w for w in re.split(r'[/.,?\- \n\t\[\]\{\}]', str(text))
            if w != '']


def _extract_text_from_node(node):
    n_type = type(node)
    if n_type == mw_nodes.ExternalLink:
        text = str(node.title)
    elif n_type == mw_nodes.Heading:
        text = str(node.title)
    elif n_type == mw_nodes.Tag:
        text = str(node.contents)
    elif n_type == mw_nodes.Text:
        text = node.value
    elif n_type == mw_nodes.Wikilink:
        if node.text is not None:
            text = str(node.text)
        else:
            text = node.title
    else:
        text = ''
    return text.strip()


def _transform(n, wikicode_list):
    result = []
    for wcode in wikicode_list:
        words = _select_first_n_words_from_wikicode(n, wcode)
        joined = ' '.join(words)
        result.append(joined)
    return result


class FeatureExtractor(object):

    def __init__(self, n):
        self._n = n
        self._vectorizer = None

    def fit_extract(self, wikicode_list):
        first_n_words = _transform(self._n, wikicode_list)
        self._vectorizer = TfidfVectorizer(min_df=0.02)
        X = self._vectorizer.fit_transform(first_n_words)
        return X.toarray()

    def extract(self, wikicode_list):
        first_n_words = _transform(self._n, wikicode_list)
        X = self._vectorizer.transform(first_n_words)
        return X.toarray()

    def _select_first_n_words_from_wikicode(self, wikicode):
        first_words = []
        for node in wikicode.nodes:
            text = _extract_text_from_node(node)
            words = _extract_words_from_text(text)
            first_words.extend(words)
            if len(first_words) > self._n:
                break
        return first_words[:self._n]
