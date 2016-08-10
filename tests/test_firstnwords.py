import unittest
import mwparserfromhell as mwp
import pageclassifier.features.firstnwords as fnw
from . import utilities as utils


class FirstNWordsTest(unittest.TestCase):

    def test_no_error_on_basic_usage(self):
        texts = utils.get_cached_revisions()
        wcode_list = [mwp.parse(text) for text in texts.values()]
        item = wcode_list.pop()
        # Reducing word frequency to allow them to persist
        wcode_list.extend([item] * (len(wcode_list) * 10))
        n = 30
        extractor = fnw.FeatureExtractor(n)

        extractor.fit_extract(wcode_list)
        extractor.extract(wcode_list)

    def test_transform(self):
        texts = ['[[Text]] here and there', '[[Text]] there and here']
        wcode_list = [mwp.parse(text) for text in texts]
        n = 2

        transformed = fnw._transform(n, wcode_list)

        self.assertEqual(len(transformed), len(wcode_list))
        self.assertEqual(transformed[0], 'Text here')
        self.assertEqual(transformed[1], 'Text there')
