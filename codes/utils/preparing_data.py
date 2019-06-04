import text_utils
import tensorflow as tf
import emoji


tf.flags.DEFINE_string("input_file", "../Data/text8.txt", "input file to pre-process")
tf.flags.DEFINE_string("output_file", "../Data/text8.txt.clean", "Output file after pre-processing")

FLAGS = tf.flags.FLAGS

data_samples = list(open(FLAGS.input_file, "r").readlines())
data_samples = [emoji.demojize(s.strip()) for s in data_samples]

x_text = [text_utils.clean_text(sent) for sent in data_samples]

file_writer = open(FLAGS.output_file, "w", encoding="utf-8")
for line in x_text:
    file_writer.write("%s\n"%line)
file_writer.close()
