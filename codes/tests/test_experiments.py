"""Regression tests: the shipped data must reproduce the numbers reported in the paper."""
import os
import unittest

import numpy as np

CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Figure 2 of the paper (codes/images/results_fig2.png).
FIG2_WORD_NODP = [0.00849802371541502, 0.00849802371541502, 0.009300162752848175, 0.010013175230566536,
                  0.04940711462450593, 0.07636204596913491, 0.07003454105793733, 0.15435934591208544,
                  0.17578405239106296]
FIG2_WORD_DP = [0.004784688995215311, 0.004784688995215311, 0.004132231404958678, 0.004132231404958678,
                0.006493506493506493, 0.024634638422734965, 0.043843297409396145, 0.08504209100465784,
                0.10467278639445513]
FIG2_CHAR_NODP = [0.8822962491567835, 0.8825348463783746, 0.8828456512801718, 0.8834435203033301,
                  0.8833895312993248, 0.8808716831755131, 0.8814099715321754, 0.8784676970297393,
                  0.879807142366003]
FIG2_CHAR_DP = [0.8878286780236411, 0.8881138432475147, 0.8880653116217662, 0.8869174435395691,
                0.8921031088629288, 0.8902628703732823, 0.8964325646257415, 0.8948389326119149,
                0.8957137160649659]


class InCodesDirTestCase(unittest.TestCase):
    """The experiment scripts use paths relative to codes/."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        os.chdir(CODES_DIR)

    def tearDown(self):
        os.chdir(self._old_cwd)


class Experiment1Test(InCodesDirTestCase):
    def test_figure2_word_level(self):
        import eval_embedding_space
        nodp, dp = eval_embedding_space.calculate_map("word_similarity", with_gold_results=True,
                                                      is_word_level=True)
        np.testing.assert_allclose(nodp, FIG2_WORD_NODP, rtol=1e-12)
        np.testing.assert_allclose(dp, FIG2_WORD_DP, rtol=1e-12)

    def test_figure2_character_level(self):
        import eval_embedding_space
        nodp, dp = eval_embedding_space.calculate_map("word_similarity", with_gold_results=True,
                                                      is_word_level=False)
        np.testing.assert_allclose(nodp, FIG2_CHAR_NODP, rtol=1e-12)
        np.testing.assert_allclose(dp, FIG2_CHAR_DP, rtol=1e-12)


class PersonalityDataTest(InCodesDirTestCase):
    def test_process_personality_data(self):
        import pandas as pd
        from utils import dataset_utils
        raw = pd.read_csv("../data/personality/Full_Data.csv", encoding="utf-8", sep=';')
        docs, labels, author_ids, sext = dataset_utils.process_personality_data(raw)
        self.assertEqual(len(docs), 250)
        # Used to be an object array of 5 lazy `map` objects under Python 3.
        self.assertEqual(labels.shape, (250, 5))
        self.assertTrue(set(np.unique(labels)) <= set(range(10)))
        self.assertEqual(sext.shape, (250,))

    def test_cached_features_load(self):
        from feature_extractors import text_features
        tf_obj = text_features.Text_Features()
        tf_obj.saved_dir = "../data/saved/saved_dp/w2v_20_epochs"
        X_train, X_test, y_train, y_test = tf_obj.feature_generator([], [], 0.2, load_from_file=True)
        self.assertEqual(X_train.shape, (200, 2202))
        self.assertEqual(X_test.shape, (50, 2202))
        self.assertEqual(y_train.shape, (200,))
        self.assertEqual(y_test.shape, (50,))


if __name__ == "__main__":
    unittest.main()
