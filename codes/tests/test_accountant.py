import unittest

try:
    import tensorflow.compat.v1 as tf
except ImportError:  # TensorFlow is only needed for training
    tf = None


@unittest.skipIf(tf is None, "TensorFlow is not installed")
class GaussianMomentsAccountantTest(unittest.TestCase):
    def test_privacy_spent_for_several_target_eps(self):
        from differential_privacy import accountant
        graph = tf.Graph()
        with graph.as_default():
            acc = accountant.GaussianMomentsAccountant(total_examples=50000)
            accum_op = acc.accumulate_privacy_spending([None, None], 5, 128)
            with tf.Session(graph=graph) as sess:
                sess.run(tf.global_variables_initializer())
                for _ in range(100):
                    sess.run(accum_op)
                target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]
                spent = acc.get_privacy_spent(sess, target_eps=target_eps)
        self.assertEqual([e.spent_eps for e in spent], target_eps)
        deltas = [e.spent_delta for e in spent]
        # Larger eps => smaller (or equal) delta. Before the zip() fix every eps after the first
        # reported delta == 1.0 because the moments iterator was already exhausted.
        self.assertEqual(deltas, sorted(deltas, reverse=True))
        self.assertLess(deltas[-1], 1e-10)


if __name__ == "__main__":
    unittest.main()
