import unittest
import mwparserfromhell as mwp
import pageclassifier.features.firstnwords as fnw


class FirstNWordsTest(unittest.TestCase):

    def test_extract_text_from_node_template(self):
        wcode = mwp.parse('{{test template | thing}}')
        node = wcode[0]

        result = fnw._extract_text_from_node(node)

        self.assertEqual(result, '')

    def test_extract_text_from_node_wikilink(self):
        wcode = mwp.parse('[[foo | Text2]]')
        node = wcode.nodes[0]

        result = fnw._extract_text_from_node(node)

        self.assertEqual(result, 'Text2')

    def test_select_first_n_words_from_wikicode(self):
        text = ('{{test template | thing}} Text1, [[foo | Text2]].'
                'text3 [[text4]] text5')
        wcode = mwp.parse(text)
        fx = fnw.FeatureExtractor(4)

        result = fx._select_first_n_words_from_wikicode(wcode)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 'Text1')
        self.assertEqual(result[1], 'Text2')
        self.assertEqual(result[2], 'text3')
        self.assertEqual(result[3], 'text4')
