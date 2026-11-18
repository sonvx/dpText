"""Normalise a raw text corpus (one document per line) before training embeddings.

Usage (from the codes/ directory):
    python3 -m utils.preparing_data --input_file ../data/text8.txt --output_file ../data/text8.txt.clean
"""
import argparse

import emoji

from utils import text_utils


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input_file", default="../data/text8.txt", help="input file to pre-process")
    parser.add_argument("--output_file", default="../data/text8.txt.clean",
                        help="Output file after pre-processing")
    args = parser.parse_args()

    with open(args.input_file, "r", encoding="utf-8") as reader:
        data_samples = [emoji.demojize(line.strip()) for line in reader]

    x_text = [text_utils.clean_text(sent) for sent in data_samples]

    with open(args.output_file, "w", encoding="utf-8") as file_writer:
        for line in x_text:
            file_writer.write("%s\n" % line)


if __name__ == "__main__":
    main()
