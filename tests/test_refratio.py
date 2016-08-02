import unittest
import mwparserfromhell as mwp
import pageclassifier.features.refratio as rr


class RefRatioTest(unittest.TestCase):

    def test_no_error_on_provided_training_set(self):
        rr.FeatureExtractor('training_data/training_refs.csv')
