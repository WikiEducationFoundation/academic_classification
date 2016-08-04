import unittest
import mwparserfromhell as mwp
import pageclassifier.features.infobox as ifb
from . import utilities as utils


class InfoboxTest(unittest.TestCase):

    def test_filter_infobox_names(self):
        wcode = mwp.parse('{{Infobox something | thing}}'
                          '{{not-one else}}'
                          '{{infobox again}}')

        result = ifb._filter_infobox_names(wcode)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'Infobox something')
        self.assertEqual(result[1], 'infobox again')

    def test_transform(self):
        wcode_list = [mwp.parse('{{Infobox something | thing}}'
                                '{{not-one else}}'
                                '{{infobox again}}'),
                      mwp.parse('{{Infobox num1 | thing}}'
                                '{{not-one else}}'
                                '{{infobox num2}}')]

        result = ifb._transform(wcode_list)

        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], 'infobox-something infobox-again')
        self.assertEqual(result[1], 'infobox-num1 infobox-num2')

    def test_no_error_on_basic_usage(self):
        texts = utils.get_cached_revisions()
        wcode_list = [mwp.parse(text) for text in texts.values()]
        extractor = ifb.FeatureExtractor()

        extractor.fit_extract(wcode_list)
        extractor.extract(wcode_list)
