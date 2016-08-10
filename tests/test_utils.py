import unittest
import mwparserfromhell as mwp
import pageclassifier.features.utils as featutils


class UtilsTest(unittest.TestCase):

    def test_extract_text_from_node_template(self):
        wcode = mwp.parse('{{test template | thing}}')
        node = wcode[0]

        result = featutils._extract_text_from_node(node)

        self.assertEqual(result, '')

    def test_extract_text_from_node_wikilink(self):
        wcode = mwp.parse('[[foo | Text2]]')
        node = wcode.nodes[0]

        result = featutils._extract_text_from_node(node)

        self.assertEqual(result, 'Text2')

    def test_select_first_n_words_from_wikicode(self):
        text = ('{{test template | thing}} Text1, [[foo | Text2]].'
                'text3 [[text4]] text5')
        wcode = mwp.parse(text)
        first_n = 4

        result = featutils.select_words_from_wikicode(wcode, first_n=first_n)

        self.assertEqual(len(result), 4)
        self.assertEqual(result[0], 'Text1')
        self.assertEqual(result[1], 'Text2')
        self.assertEqual(result[2], 'text3')
        self.assertEqual(result[3], 'text4')
