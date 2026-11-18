"""Build the dashboard's word-level sample embedding from data/text8.zip.

A gensim skip-gram word2vec (200 dimensions, the 50,000 most frequent words, 3 epochs) trained
without differential privacy on the same text8 corpus as the paper. It is written to
data/embeddings/text8_gensim_skipgram_200d.bin (word2vec binary), where the dashboard picks it up
as a project sample. Takes about a minute on a laptop.

    cd codes && python -m dashboard.make_sample_embeddings
"""
import argparse
import os
import time
import zipfile

from dashboard import embedding_sources as es

TEXT8_ZIP = os.path.join(es.DATA_DIR, "text8.zip")


def read_text8(max_tokens=None):
    with zipfile.ZipFile(TEXT8_ZIP) as zf:
        words = zf.read(zf.namelist()[0]).decode("utf-8").split()
    return words[:max_tokens] if max_tokens else words


def build_text8_sample(out_path=None, epochs=3, max_tokens=None, max_vocab=50000, workers=None):
    """Train the sample and save it; returns (out_path, number of words, seconds)."""
    from gensim.models import Word2Vec

    out_path = out_path or es.text8_sample_path()
    words = read_text8(max_tokens)
    sentences = [words[i:i + 1000] for i in range(0, len(words), 1000)]
    started = time.time()
    model = Word2Vec(sentences, vector_size=200, window=5, min_count=5, max_final_vocab=max_vocab, sg=1,
                     negative=10, workers=workers or os.cpu_count() or 1, epochs=epochs, seed=42)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    tmp = out_path + ".part"
    model.wv.save_word2vec_format(tmp, binary=True)
    os.replace(tmp, out_path)
    return out_path, len(model.wv), time.time() - started


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=None, help="output path (default: data/embeddings/%s)" % es.TEXT8_SAMPLE_NAME)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()
    path, n_words, seconds = build_text8_sample(args.out, epochs=args.epochs)
    print("Trained in %.0f s: %d words -> %s" % (seconds, n_words, path))


if __name__ == "__main__":
    main()
