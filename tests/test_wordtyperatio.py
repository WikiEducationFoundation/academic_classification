import unittest
import mwparserfromhell as mwp
import pageclassifier.features.wordtyperatio as wtr
import mwparserfromhell as mwp
from . import utilities as utils

wordtype_filepaths = [
    'resources/academic_words_sl01.csv',
    'resources/academic_words_sl02.csv'
]


class WordTypeRatioTest(unittest.TestCase):

    def test_no_error_on_init(self):
        wtr.FeatureExtractor(wordtype_filepaths=wordtype_filepaths)

    def test_no_error_on_basic_usage(self):
        texts = utils.get_cached_revisions()
        wcode_list = [mwp.parse(text) for text in texts.values()]
        extractor = wtr.FeatureExtractor(wordtype_filepaths=wordtype_filepaths)

        extractor.fit_extract(wcode_list)
        extractor.extract(wcode_list)
