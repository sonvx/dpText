"""Train word2vec (skip-gram) embeddings on text8 with or without differential privacy.

The model is written in TensorFlow 1.x graph style and runs on TensorFlow 2.x through
``tf.compat.v1`` (v2 behaviour disabled), so the training algorithm is unchanged.

Usage (from the codes/ directory, see 10.run_train_dp_embedding.sh):
    PYTHONPATH=$PWD python3 ./word2vec/tf_dp_word_embedding.py --train_path=../data/text8.zip \
        --trained_models=../data/text8_data/trained_models_dp_dpsgd --with_dp=True --clip_by_norm=True
"""
import collections
import json
import logging
import math
import os
import random
import zipfile

import numpy as np
import tensorflow.compat.v1 as tf
from absl import app, flags

# See good outputs from Tex8
# here: https://github.com/tensorflow/tensorflow/blob/master/tensorflow/examples/udacity/5_word2vec.ipynb
from differential_privacy import accountant
from differential_privacy import utils as dp_utils
from utils import file_utils
from word2vec import embedding_utils

tf.disable_v2_behavior()

flags.DEFINE_string("train_path", "../data/text8.zip",
                    "Training file path. E.g., ../data/text8.zip")
flags.DEFINE_bool("nce_loss", False,
                  "Unused, kept for backward compatibility. Use --with_nce_loss.")
flags.DEFINE_string("trained_models", "../data/text8_data/trained_models",
                    "Trained models path.")
flags.DEFINE_integer("NUM_STEPS", 100001,
                     "Number of running epochs.")
flags.DEFINE_bool("with_dp", False,
                  "Train with differential privacy guarantee")

flags.DEFINE_bool("RESTORE_LAST_CHECK_POINT", False,
                  "Restore last check point and keep training up to the new limited epochs.")

flags.DEFINE_bool("DEBUG", False, "Enable debug.")

flags.DEFINE_bool("with_nce_loss", False, "Using nce_loss instead of softmax loss.")
flags.DEFINE_string("optimizer", "sgd", "Optimizer: sgd by default. adam is another option.")
flags.DEFINE_bool("save_best_model_alltime", False, "Save best model everytime we see it. Warning: super slow.")
flags.DEFINE_bool("clip_by_norm", False, "Clip by norm is faster than clip by Abadi et al. in DP-SGD")

FLAGS = flags.FLAGS

# Set in main(); used as module-level state by the helpers below.
data = None
data_index = 0
reverse_dictionary = None
vocabulary_size = 50000
num_steps = None
max_early_stopping = -1
target_eps = None
saving_pointers = None


# def maybe_download(filename, expected_bytes):
#     """Download a file if not present, and make sure it's the right size."""
#     if not os.path.exists(filename):
#         filename, _ = urlretrieve(url + filename, filename)
#     statinfo = os.stat(filename)
#     if statinfo.st_size == expected_bytes:
#         print('Found and verified %s' % filename)
#     else:
#         print(statinfo.st_size)
#         raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
#     return filename


def collect_data(vocabulary_size=10000, test_data=False, reload_from_files=False):
    # if test_data:
    #     url = 'http://mattmahoney.net/dc/'
    #     filename = maybe_download(FLAGS.train_path, url, 31344016)
    # else:
    filename = FLAGS.train_path
    vocabulary = read_data(filename)
    print(vocabulary[:7])
    data, count, dictionary, reverse_dictionary = build_dataset(vocabulary,
                                                                vocabulary_size)
    del vocabulary  # Hint to reduce memory.
    return data, count, dictionary, reverse_dictionary


def read_data(filename):
    file_out = filename + "_saved"

    if not os.path.exists(file_out + ".pkl"):
        """Extract the first file enclosed in a zip file as a list of words."""
        with zipfile.ZipFile(filename) as f:
            data = f.read(f.namelist()[0]).decode("utf-8").split()

            # Not recommended. Please use preparing_data.py instead.
            # if not FLAGS.test_data: # only clean on real data
            #     for index, item in enumerate(data):
            #         data[index] = text_utils.clean_str(item)

        # saving this to file
        file_utils.save_obj(data, file_out)
    else:
        # restore it
        data = file_utils.load_obj(file_out)

    return data


def build_dataset(words, n_words):
    """Process raw inputs into a dataset."""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words - 1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary


def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    buffer = collections.deque(maxlen=span)
    # random.shuffle(data)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    for i in range(batch_size // num_skips):
        target = skip_window  # target label at the center of the buffer
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            while target in targets_to_avoid:
                target = random.randint(0, span - 1)
            targets_to_avoid.append(target)
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)
    return batch, labels


# Step 6: Visualize the embeddings.
def plot_with_labels(low_dim_embs, labels, filename):
    """
    # pylint: disable=missing-docstring
    # Function to draw visualization of distance between embeddings.
    :param low_dim_embs:
    :param labels:
    :param filename:
    :return:
    """
    import matplotlib.pyplot as plt
    plt.switch_backend('agg')

    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'
    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(
            label,
            xy=(x, y),
            xytext=(5, 2),
            textcoords='offset points',
            ha='right',
            va='bottom')

    plt.savefig(filename)
    plt.close()  # free the figure; this is called at every saving point


def saving_state(dir_out_path, spent_eps_deltas, temp_embeddings, tf_saver, tf_session):

    """
    Saving states of embedding
    :param dir_out_path:
    :param spent_eps_deltas:
    :param normalized_embeddings:
    :return:
    """
    # Save embedding models.

    log_dir = os.path.join(dir_out_path, "logs")
    # dir_out_path may already exist (e.g. the "_best_one" folder), so always ensure log_dir exists.
    os.makedirs(log_dir, exist_ok=True)

    tf_saver.save(tf_session, os.path.join(log_dir, 'model.ckpt'))
    with open(os.path.join(dir_out_path, "manual.logs"), "w") as writer:
        writer.write("%s" % spent_eps_deltas)
    embedding_utils.save_embedding_models_tofolder(dir_out_path, temp_embeddings, reverse_dictionary, vocabulary_size)

    try:
        # pylint: disable=g-import-not-at-top
        from sklearn.manifold import TSNE
        # final_embeddings = temp_embeddings.eval()
        tsne = TSNE(
            perplexity=30, n_components=2, init='pca', max_iter=5000, method='exact')
        plot_only = 500
        low_dim_embs = tsne.fit_transform(temp_embeddings[:plot_only, :])
        labels = [reverse_dictionary[i] for i in range(plot_only)]
        print("Wrote to ", dir_out_path)
        plot_with_labels(low_dim_embs, labels, os.path.join(dir_out_path, 'tsne.png'))

    except ImportError as ex:
        print('Please install sklearn, matplotlib, and scipy to show embeddings.')
        print(ex)


def training_embedding(reverse_dictionary, with_dp = False):
    """
    # training with DP
    :param with_dp:
    :return:
    """
    batch_size = 128
    embedding_size = 300  # Dimension of the embedding vector.
    skip_window = 1  # How many words to consider left and right.
    num_skips = 2  # How many times to reuse an input to generate a label.
    # We pick a random validation set to sample nearest neighbors. here we limit the
    # validation samples to the words that have a low numeric ID, which by
    # construction are also the most frequent.
    valid_size = 16  # Random set of words to evaluate similarity on.
    valid_window = 100  # Only pick dev samples in the head of the distribution.
    valid_examples = np.array(random.sample(range(valid_window), valid_size))
    num_sampled = 64  # Number of negative examples to sample.

    learning_rate = 1

    # DP parameters
    clip_bound = 0.01  # 'the clip bound of the gradients'
    # num_steps = 160000  # 'number of steps T = E * N / L = E / q'
    sigma = 5  # 'sigma'
    delta = 1e-5  # 'delta'

    # allow_soft_placement: fall back to the CPU for the ops pinned to '/gpu:0' below when no GPU
    # is available (otherwise TensorFlow raises an error on CPU-only machines).
    sess = tf.InteractiveSession(config=tf.ConfigProto(allow_soft_placement=True))

    # with graph.as_default(), tf.device('/cpu:0'):
    # Input data.
    with tf.device('/gpu:0'):
        train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
        train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

        # Variables.
        embeddings = tf.Variable(
            tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))

    # Model.
    # Look up embeddings for inputs.
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)

    if FLAGS.with_nce_loss:
        nce_weights = tf.Variable(
            tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
        nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

        cross_entropy = tf.reduce_mean(
                        tf.nn.nce_loss(
                            weights=nce_weights,
                            biases=nce_biases,
                            labels=train_labels,
                            inputs=embed,
                            num_sampled=num_sampled,
                            num_classes=vocabulary_size))
    else:
        with tf.device('/gpu:0'):
            softmax_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size],
                                stddev=1.0 / math.sqrt(embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([vocabulary_size]))
            # Compute the softmax loss, using a sample of the negative labels each time.
            # Read more: https://stackoverflow.com/questions/37671974/tensorflow-negative-sampling
            # When we want to compute the softmax probability for your true label,
            # we compute: logits[true_label] / sum(logits[negative_sampled_labels]
            # Other candidate sampling: https://www.tensorflow.org/extras/candidate_sampling.pdf
        cross_entropy = tf.reduce_mean(
            tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                       labels=train_labels, num_sampled=num_sampled, num_classes=vocabulary_size))

    priv_accountant = accountant.GaussianMomentsAccountant(vocabulary_size)
    privacy_accum_op = priv_accountant.accumulate_privacy_spending([None, None], sigma, batch_size)

    # Optimizer.
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    # optimizer = tf.train.AdagradOptimizer(learning_rate).minimize(loss)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cross_entropy)

    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    if FLAGS.optimizer == "adam":
        # cannot use adam so far. Tested and the model couldn't converge.
        optimizer = tf.train.AdamOptimizer(learning_rate)
        print("##INFO: Using adam optimizer")
    if FLAGS.optimizer == "adagrad":
        # cannot use adam so far. Tested and the model couldn't converge.
        optimizer = tf.train.AdagradOptimizer(learning_rate)
        print("##INFO: Using adagrad optimizer")

    log_dir = os.path.join(FLAGS.trained_models, "logs")

    # compute gradient
    if FLAGS.with_nce_loss:
        gw_Embeddings = tf.gradients(cross_entropy, embeddings)[0]  # gradient of embeddings
        gw_softmax_weights = tf.gradients(cross_entropy, nce_weights)[0]  # gradient of nce_weights
        gb_softmax_biases = tf.gradients(cross_entropy, nce_biases)[0]  # gradient of nce_biases
    else:
        with tf.device('/gpu:0'):
            gw_Embeddings = tf.gradients(cross_entropy, embeddings)[0]  # gradient of embeddings
            gw_softmax_weights = tf.gradients(cross_entropy, softmax_weights)[0]  # gradient of softmax_weights
            gb_softmax_biases = tf.gradients(cross_entropy, softmax_biases)[0]  # gradient of softmax_biases

    # clip gradient
    if FLAGS.clip_by_norm:
        # faster but takes more epochs to train
        with tf.device('/gpu:0'):
            gw_Embeddings = tf.clip_by_norm(gw_Embeddings, clip_bound)
            gw_softmax_weights = tf.clip_by_norm(gw_softmax_weights, clip_bound)
            gb_softmax_biases = tf.clip_by_norm(gb_softmax_biases, clip_bound)
    else:
        # dp-sgd: slow and require more memory but converge faster, take less epochs.
        gw_Embeddings = dp_utils.BatchClipByL2norm(gw_Embeddings, clip_bound)
        gw_softmax_weights = dp_utils.BatchClipByL2norm(gw_softmax_weights, clip_bound)
        gb_softmax_biases = dp_utils.BatchClipByL2norm(gb_softmax_biases, clip_bound)

    sensitivity = clip_bound  # adjacency matrix with one more tuple

    # Add noise
    if FLAGS.with_dp:
        gw_Embeddings += tf.random_normal(shape=tf.shape(gw_Embeddings),
                                                mean=0.0, stddev=sigma * (sensitivity ** 2), dtype=tf.float32)
        gw_softmax_weights += tf.random_normal(shape=tf.shape(gw_softmax_weights),
                                                mean=0.0, stddev=sigma * (sensitivity ** 2), dtype=tf.float32)
        gb_softmax_biases += tf.random_normal(shape=tf.shape(gb_softmax_biases),
                                                mean=0.0, stddev=sigma * (sensitivity ** 2), dtype=tf.float32)

    if FLAGS.with_nce_loss:
        train_step = optimizer.apply_gradients([(gw_Embeddings, embeddings),
                                                (gw_softmax_weights, nce_weights),
                                                (gb_softmax_biases, nce_biases)])
    else:
        train_step = optimizer.apply_gradients([(gw_Embeddings, embeddings),
                                            (gw_softmax_weights, softmax_weights),
                                            (gb_softmax_biases, softmax_biases)])

    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:
    with tf.device('/gpu:0'):
        norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))
    # Built once here: creating it inside the training loop kept adding new ops to the graph.
    mean_abs_embedding = tf.reduce_mean(tf.abs(embeddings))

    min_loss = 10**4
    per_dec_count = 0

    print('Initialized')
    average_loss = 0

    running = True
    step = 0
    average_loss_arr = []
    saving_pointer_idx = 0
    current_avg_loss = None
    spent_eps_deltas = None

    # put it here because Adam has its own variables.
    sess.run(tf.global_variables_initializer())

    # saver must be used after global_variables_initializer
    saver = tf.train.Saver()

    # Save the variables to disk.
    save_path = os.path.join(FLAGS.trained_models, "initialized_model.ckpt")
    # Sonvx: we need to make sure initialized variables are all the same for different tests.
    print("Checking on path: ", save_path)
    if not os.path.isfile(save_path + ".index"):
        saved_info = saver.save(sess, save_path)
        print("Global initialized model saved in file: %s" % saved_info)
    else:
        saver.restore(sess, save_path)
        print("Restored the global initialized model.")
    if FLAGS.DEBUG:
        input("Double check whether or not the initialized model got restored then <Press enter>")
    print('###INFO: Initialized in run(graph)')

    if FLAGS.RESTORE_LAST_CHECK_POINT:
        checkpoint_path = os.path.join(log_dir, "model.ckpt")
        if os.path.isfile(checkpoint_path + ".index"):
            saver.restore(sess, checkpoint_path)
            print("Restored the latest checkpoint at %s." % (checkpoint_path))

    # note that computing similarity is expensive (~20% slowdown if computed every 500 steps).
    # int(): with the default NUM_STEPS=100001 the old float (20000.2) only matched step 0.
    check_step = max(1, int(FLAGS.NUM_STEPS * 0.2))

    while running:
        # for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        if FLAGS.DEBUG:
            print("Global data_index = ", data_index)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}

        # old: sess.run([optimizer, cross_entropy], feed_dict=feed_dict)
        # template: train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: 0.5});
        train_step.run(feed_dict=feed_dict)
        loss = cross_entropy.eval(feed_dict=feed_dict)

        sess.run([privacy_accum_op])
        # print(step, spent_eps_deltas)

        average_loss += loss

        if step == 0:
            step_dev = 0.1 * 5
        else:
            step_dev = step

        current_avg_loss = np.mean(average_loss) / step_dev
        average_loss_arr.append(current_avg_loss)

        if step % 200 == 0:
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, current_avg_loss))
            print("Embedding: ")
            print(sess.run(mean_abs_embedding))

        if step % check_step == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)

        current_saving_dir = os.path.join(FLAGS.trained_models, "_%sepoch" %
                                          (saving_pointers[saving_pointer_idx]))
        # EARLY STOPPING
        if min_loss >= current_avg_loss:
            min_loss = current_avg_loss
            per_dec_count = 0

            if FLAGS.save_best_model_alltime:
                best_of_saving_point_dir = os.path.join(current_saving_dir, "_best_one")
                os.makedirs(best_of_saving_point_dir, exist_ok=True)

                temp_embeddings = normalized_embeddings.eval()
                spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
                saving_state(best_of_saving_point_dir, spent_eps_deltas, temp_embeddings, saver, sess)
            msg = ("Got best model so far at step %s , avg loss = %s"%(step, current_avg_loss))
            logging.info(msg)
            print (msg)
        else:
            per_dec_count += 1

        step += 1

        if per_dec_count == max_early_stopping or step == num_steps:
            running = False

        if (step + 1) in saving_pointers:
            spent_eps_deltas = priv_accountant.get_privacy_spent(sess, target_eps=target_eps)
            folder_path = os.path.join(FLAGS.trained_models, "_%sepoch" % (step + 1))
            temp_embeddings = normalized_embeddings.eval()
            saving_state(folder_path, spent_eps_deltas, temp_embeddings, saver, sess)
            # Make sure we don't increase saving_pointer_idx larger than what the total number of pointers we set.
            if saving_pointer_idx < len(saving_pointers) -1:
                saving_pointer_idx += 1
            msg = "##INFO: STEP %s: avg_loss history: avg_loss_arr = %s" % (step, average_loss_arr)
            logging.info(msg)

        if num_steps > 1 and step % (num_steps - 1) == 0:
            print("Final privacy spent: ", step, spent_eps_deltas)

    print("Stopped at %s, \nFinal avg_loss = %s" % (step, current_avg_loss))

    # final_embeddings = normalized_embeddings.eval()
    sess.close()


def main(argv):
    del argv  # unused
    global data, data_index, reverse_dictionary, vocabulary_size, num_steps, max_early_stopping
    global target_eps, saving_pointers

    os.makedirs(FLAGS.trained_models, exist_ok=True)
    # force=True: absl installs its own root handler, which would otherwise make this a no-op.
    logging.basicConfig(filename=os.path.join(FLAGS.trained_models, "manual.logs"), filemode="w",
                        level=logging.DEBUG, force=True)
    logging.info(json.dumps(FLAGS.flag_values_dict()))

    # num_steps = 10000  # 'number of steps T = E * N / L = E / q' (epoch * total_docs/ lot_size).
    num_steps = FLAGS.NUM_STEPS
    max_early_stopping = -1 # num_steps / 10
    target_eps = [0.125, 0.25, 0.5, 1, 2, 4, 8]

    saving_pointers = [20, 200, 500, 1000, 5000, 10000, 20000, 50000, 90000, num_steps-1]

    # url = 'http://mattmahoney.net/dc/'
    # filename = maybe_download('../Data/text8.zip', 31344016)
    # words = read_data(filename)
    # print('Data size %d' % len(words))

    vocabulary_size = 50000
    data, count, dictionary, reverse_dictionary = collect_data(vocabulary_size=vocabulary_size)

    # data, count, dictionary, reverse_dictionary = build_dataset(words)
    print('Most common words (+UNK)', count[:5])
    print('Sample data', data[:10])
    # del words  # Hint to reduce memory.
    data_index = 0

    print('data:', [reverse_dictionary[di] for di in data[:8]])

    for num_skips, skip_window in [(2, 1), (4, 2)]:
        data_index = 0
        batch, labels = generate_batch(batch_size=8, num_skips=num_skips, skip_window=skip_window)
        print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
        print('    batch:', [reverse_dictionary[bi] for bi in batch])
        print('    labels:', [reverse_dictionary[li] for li in labels.reshape(8)])

    training_embedding(reverse_dictionary, with_dp=FLAGS.with_dp)


if __name__ == "__main__":
    app.run(main)
