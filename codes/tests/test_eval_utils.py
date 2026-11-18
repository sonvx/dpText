import unittest

from utils import eval_utils


class ApkTest(unittest.TestCase):
    """Average precision @k (ported from the old Python 2 module-level checks)."""

    def test_apk(self):
        self.assertAlmostEqual(eval_utils.apk(list(range(1, 6)), [6, 4, 7, 1, 2], 2), 0.25)
        self.assertAlmostEqual(eval_utils.apk(list(range(1, 6)), [1, 1, 1, 1, 1], 5), 0.2)
        predicted = list(range(1, 21)) + list(range(200, 600))
        self.assertAlmostEqual(eval_utils.apk(list(range(1, 100)), predicted, 20), 1.0)

    def test_apk_empty_actual(self):
        self.assertEqual(eval_utils.apk([], [1, 2, 3], 3), 0.0)


class MapTest(unittest.TestCase):
    def test_word_level(self):
        self.assertAlmostEqual(eval_utils.calc_map(["a", "b", "c"], ["a", "b", "c"], topK=3), 1.0)
        # hits at ranks 1 and 3 -> mean(1/1, 2/3)
        self.assertAlmostEqual(eval_utils.calc_map(["a", "b"], ["a", "x", "b"], topK=3), (1 + 2 / 3) / 2)
        self.assertEqual(eval_utils.calc_map(["a"], ["x", "y"], topK=2), 0.0)

    def test_character_level_no_hits_is_zero_not_nan(self):
        self.assertEqual(eval_utils.calc_map_character_level(["ab"], ["xy"], topK=1), 0.0)

    def test_mapk_dispatch(self):
        self.assertEqual(eval_utils.mapk(["a"], ["a"], k=1, word_level=True),
                         eval_utils.calc_map(["a"], ["a"], topK=1))
        self.assertEqual(eval_utils.mapk(["ab"], ["ab"], k=1, word_level=False),
                         eval_utils.calc_map_character_level(["ab"], ["ab"], topK=1))


if __name__ == "__main__":
    unittest.main()
