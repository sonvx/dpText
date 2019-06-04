# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Contains utilities for downloading and converting datasets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import tarfile
import re

from six.moves import urllib
import tensorflow as tf
from collections import Counter
import itertools
import pandas as pd

LABELS_FILENAME = 'labels.txt'


def int64_feature(values):
    """Returns a TF-Feature of int64s.

    Args:
      values: A scalar or list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(int64_list=tf.train.Int64List(value=values))


def bytes_feature(values):
    """Returns a TF-Feature of bytes.

    Args:
      values: A string.

    Returns:
      A TF-Feature.
    """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))


def float_feature(values):
    """Returns a TF-Feature of floats.

    Args:
      values: A scalar of list of values.

    Returns:
      A TF-Feature.
    """
    if not isinstance(values, (tuple, list)):
        values = [values]
    return tf.train.Feature(float_list=tf.train.FloatList(value=values))


def image_to_tfexample(image_data, image_format, height, width, class_id):
    return tf.train.Example(features=tf.train.Features(feature={
        'image/encoded': bytes_feature(image_data),
        'image/format': bytes_feature(image_format),
        'image/class/label': int64_feature(class_id),
        'image/height': int64_feature(height),
        'image/width': int64_feature(width),
    }))


def download_and_uncompress_tarball(tarball_url, dataset_dir):
    """Downloads the `tarball_url` and uncompresses it locally.

    Args:
      tarball_url: The URL of a tarball file.
      dataset_dir: The directory where the temporary files are stored.
    """
    filename = tarball_url.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading %s %.1f%%' % (
            filename, float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()

    filepath, _ = urllib.request.urlretrieve(tarball_url, filepath, _progress)
    print()
    statinfo = os.stat(filepath)
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    tarfile.open(filepath, 'r:gz').extractall(dataset_dir)


def write_label_file(labels_to_class_names, dataset_dir,
                     filename=LABELS_FILENAME):
    """Writes a file with the list of class names.

    Args:
      labels_to_class_names: A map of (integer) labels to class names.
      dataset_dir: The directory in which the labels file should be written.
      filename: The filename where the class names are written.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'w') as f:
        for label in labels_to_class_names:
            class_name = labels_to_class_names[label]
            f.write('%d:%s\n' % (label, class_name))


def has_labels(dataset_dir, filename=LABELS_FILENAME):
    """Specifies whether or not the dataset directory contains a label map file.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      `True` if the labels file exists and `False` otherwise.
    """
    return tf.gfile.Exists(os.path.join(dataset_dir, filename))


def read_label_file(dataset_dir, filename=LABELS_FILENAME):
    """Reads the labels file and returns a mapping from ID to class name.

    Args:
      dataset_dir: The directory in which the labels file is found.
      filename: The filename where the class names are written.

    Returns:
      A map from a label (integer) to class name.
    """
    labels_filename = os.path.join(dataset_dir, filename)
    with tf.gfile.Open(labels_filename, 'rb') as f:
        lines = f.read().decode()
    lines = lines.split('\n')
    lines = filter(None, lines)

    labels_to_class_names = {}
    for line in lines:
        index = line.index(':')
        labels_to_class_names[int(line[:index])] = line[index + 1:]
    return labels_to_class_names


def refine_text(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()


def read_sentiment_data(data_loc):
    if not data_loc.endswith("/"):
        data_loc += "/"
    file1 = data_loc + 'rt-polarity.pos'
    file2 = data_loc + 'rt-polarity.neg'
    positive_examples = list(open(file1).readlines())
    negative_examples = list(open(file2).readlines())

    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [refine_text(sent) for sent in x_text]
    x_text = [s.split(" ") for s in x_text]
    print("##INFO: Vocab_len=", len(x_text))
    # Build vocabulary
    word_counts = Counter(itertools.chain(*x_text))
    # Mapping from index to word
    vocabulary_inv = [x[0] for x in word_counts.most_common()]
    # Mapping from word to index
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
    return vocabulary


def process_personality_data(raw_data_df, is_text_label=False):
    """
    Loading personality dataset
    :param raw_data_df:
    # ;AUTHID;STATUS;sEXT;sNEU;sAGR;sCON;sOPN;cEXT;cNEU;cAGR;cCON;cOPN;DATE;NETWORKSIZE;BETWEENNESS;
    # NBETWEENNESS;DENSITY;BROKERAGE;NBROKERAGE;TRANSITIVITY;Linear_Gold_Regression
    :param is_text_label:
    :return:
    """
    import numpy as np
    # x.tolist() if
    data_df = raw_data_df.groupby("AUTHID", as_index=False).agg(lambda x: x.tolist())
    #    AUTHID                       STATUS    cEXT    cNEU    cAGR    cCON    cOPN
    # 0  usr001  [love you, do  u?, really.]  [y, y]  [n, n]  [y, y]  [n, n]  [n, n]
    # 1  usr002                     [crazy.]     [y]     [n]     [n]     [n]     [y]
    list_document = ['\n'.join(text) for text in data_df.STATUS]
    print('Df_shape=', data_df.shape, ', list_document_len=', len(list_document))

    # list of score values
    sEXT_y_data = list(map(lambda x: float(x), [y[0] for y in data_df.sEXT]))
    sOPN_y_data = list(map(lambda x: float(x), [y[0] for y in data_df.sOPN]))
    sAGR_y_data = list(map(lambda x: float(x), [y[0] for y in data_df.sAGR]))
    sCON_y_data = list(map(lambda x: float(x), [y[0] for y in data_df.sCON]))
    sNEU_y_data = list(map(lambda x: float(x), [y[0] for y in data_df.sNEU]))

    print("sEXT_y_data (%s) = \n %s [...]"%(type(sEXT_y_data), sEXT_y_data[:5]))

    print("data_df = ", data_df.shape)
    print("len_sEXT = ", len(sEXT_y_data))
    data_df['sEXT'] = sEXT_y_data
    data_df['sOPN'] = sOPN_y_data
    data_df['sAGR'] = sAGR_y_data
    data_df['sCON'] = sCON_y_data
    data_df['sNEU'] = sNEU_y_data

    # list of labels
    cEXT_y_data = map(lambda x: 0 if (x == 'y') else 1, [y[0] for y in data_df.cEXT])
    cOPN_y_data = map(lambda x: 2 if (x == 'y') else 3, [y[0] for y in data_df.cOPN])
    cAGR_y_data = map(lambda x: 4 if (x == 'y') else 5, [y[0] for y in data_df.cAGR])
    cCON_y_data = map(lambda x: 6 if (x == 'y') else 7, [y[0] for y in data_df.cCON])
    cNEU_y_data = map(lambda x: 8 if (x == 'y') else 9, [y[0] for y in data_df.cNEU])

    if is_text_label:
        cEXT_y_data = map(lambda x: 'Y.cEXT' if (x == 'y') else 'N.cEXT', [y[0] for y in data_df.cEXT])
        cOPN_y_data = map(lambda x: 'Y.cOPN' if (x == 'y') else 'N.cOPN', [y[0] for y in data_df.cOPN])
        cAGR_y_data = map(lambda x: 'Y.cAGR' if (x == 'y') else 'N.cAGR', [y[0] for y in data_df.cAGR])
        cCON_y_data = map(lambda x: 'Y.cCON' if (x == 'y') else 'N.cCON', [y[0] for y in data_df.cCON])
        cNEU_y_data = map(lambda x: 'Y.cNEU' if (x == 'y') else 'N.cNEU', [y[0] for y in data_df.cNEU])


    labels_list = np.array([cEXT_y_data, cOPN_y_data, cAGR_y_data, cCON_y_data, cNEU_y_data])
    labels_list = np.transpose(labels_list)

    print('Type(cEXT_y_data)=', type(cEXT_y_data), ', labels_list_size=', len(labels_list))

    return list_document, labels_list, data_df.AUTHID.values, data_df.sEXT.values
