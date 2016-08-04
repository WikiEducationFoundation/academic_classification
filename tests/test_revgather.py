import unittest
import pageclassifier.revgather as rg
import warnings


class RevGatherTest(unittest.TestCase):

    def test_get_text_for_revision(self):
        warnings.simplefilter("ignore", ResourceWarning)
        revids = range(100000, 100200)

        revs = rg.get_text_for_revisions(revids)

        self.assertEqual(len(revids), len(revs))
