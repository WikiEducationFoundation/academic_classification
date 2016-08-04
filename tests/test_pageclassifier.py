import unittest
import pageclassifier.pageclassifier as pc
import mwparserfromhell as mwp
from . import utilities as utils
import pageclassifier.features.refratio as rr
import pageclassifier.features.firstnwords as fnw
import pageclassifier.features.infobox as ifb


class PageClassifierTest(unittest.TestCase):

    def test_no_error_on_provided_training_set(self):
        rr.FeatureExtractor('training_data/training_refs.csv')

    def test_no_error_on_basic_usage(self):
        extractors = [rr.FeatureExtractor('training_data/training_refs.csv'),
                      fnw.FeatureExtractor(n=30),
                      ifb.FeatureExtractor()]
        clf = pc.PageClassifier(extractors)
        texts = utils.get_cached_revisions()
        wcode_list = [mwp.parse(text) for text in texts.values()]
        labels = [(i % 2 == 0) for i in texts]

        clf.train(wcode_list, labels)
        clf.predict(wcode_list[1:])
