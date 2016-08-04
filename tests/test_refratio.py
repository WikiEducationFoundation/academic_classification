import unittest
import mwparserfromhell as mwp
import pageclassifier.features.refratio as rr
import mwparserfromhell as mwp
from . import utilities as utils


class RefRatioTest(unittest.TestCase):

    def test_no_error_on_provided_training_set(self):
        rr.FeatureExtractor('training_data/training_refs.csv')

    def test_no_error_on_basic_usage(self):
        texts = utils.get_cached_revisions()
        wcode_list = [mwp.parse(text) for text in texts.values()]
        extractor = rr.FeatureExtractor('training_data/training_refs.csv')

        extractor.fit_extract(wcode_list)
        extractor.extract(wcode_list)
