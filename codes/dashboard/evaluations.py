"""Evaluations the dashboard runs on loaded embeddings, following the paper's two experiments.

* Semantic space (Figure 2): top-100 neighbours of 11 test words, scored with MAP@100 against the
  gold lists in data/evaluate_embeddings/topsimilarity100/, at word and character level.
* Personality regression (Table 3): n-gram + TF-IDF + embedding features for the myPersonality
  statuses, 80/20 split (random_state=42), linear regression and (optionally) linear SVR, RMSE.
"""
import contextlib
import io
import os

import numpy as np
import pandas as pd
from sklearn import linear_model, svm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split

from dashboard.embedding_sources import DATA_DIR, DEFAULT_CHAR_EMBEDDING
from utils import eval_utils

TOPN = 100
TESTED_WORDS = ['been', 'time', 'over', 'not', 'three', 'between', 'from', 'eight', 'they', 'more', 'other']
LEARNING_STEPS = ["20", "200", "500", "1000", "5000", "10000", "50000", "90000", "100000"]

TOPSIM_DIR = os.path.join(DATA_DIR, "evaluate_embeddings", "topsimilarity100")
GOLD_CSV = os.path.join(TOPSIM_DIR, "topsimilarity_100_gold.csv")
DP_CSV = os.path.join(TOPSIM_DIR, "dp_w2v_models.csv")
NODP_CSV = os.path.join(TOPSIM_DIR, "nodp_w2v_models.csv")
PERSONALITY_CSV = os.path.join(DATA_DIR, "personality", "Full_Data.csv")
SAVED_FEATURES_DIR = os.path.join(DATA_DIR, "saved")

TOP_COLUMNS = ["top_%d" % (i + 1) for i in range(TOPN)]


# --------------------------------------------------------------------------- semantic space (Fig. 2)


def _top_lists(csv_path):
    """{model_name: {word: [top_1 .. top_100]}} from one of the shipped top-similarity CSVs."""
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False)
    lists = {}
    for _, row in df.iterrows():
        lists.setdefault(str(row["model_name"]), {})[row["word"]] = [row[c] for c in TOP_COLUMNS]
    return lists


def gold_lists():
    """The gold top-100 neighbours of each test word (word2vec trained on text8 without DP)."""
    return next(iter(_top_lists(GOLD_CSV).values()))


def score_against_gold(predicted, gold, word_level):
    if word_level:
        return float(eval_utils.calc_map(gold, predicted, topK=TOPN))
    return float(eval_utils.calc_map_character_level(gold, predicted, topK=TOPN))


def paper_reference_curves():
    """MAP@100 of the paper's DP / non-DP models at each learning step (data behind Figure 2)."""
    gold = gold_lists()
    rows = []
    for label, csv_path in [("DP (paper)", DP_CSV), ("Non-DP (paper)", NODP_CSV)]:
        for step, word_lists in _top_lists(csv_path).items():
            for level in ("word", "character"):
                scores = [score_against_gold(word_lists[w], gold[w], level == "word") for w in TESTED_WORDS]
                rows.append({"model": label, "learning_step": step, "level": level,
                             "map": float(np.mean(scores))})
    return pd.DataFrame(rows)


def lookup_word(kv, word):
    """Return the key used for ``word`` in kv (exact match first, then lowercase), or None."""
    if word in kv.key_to_index:
        return word
    if word.lower() in kv.key_to_index:
        return word.lower()
    return None


def nearest_neighbors(kv, word, topn=10):
    key = lookup_word(kv, word)
    if key is None:
        return None
    return kv.most_similar(key, topn=topn)


def semantic_space_scores(kv, words=TESTED_WORDS, topn=TOPN):
    """Per-word MAP@100 of an embedding's neighbours against the gold lists, plus its top-100 lists."""
    gold = gold_lists()
    rows, neighbours = [], {}
    for word in words:
        key = lookup_word(kv, word)
        if key is None:
            rows.append({"word": word, "in_vocabulary": False, "map_word": 0.0, "map_character": 0.0,
                         "shared_with_gold": 0})
            continue
        predicted = [w for w, _ in kv.most_similar(key, topn=topn)]
        neighbours[word] = predicted
        rows.append({
            "word": word,
            "in_vocabulary": True,
            "map_word": score_against_gold(predicted, gold[word], word_level=True),
            "map_character": score_against_gold(predicted, gold[word], word_level=False),
            "shared_with_gold": len(set(predicted) & set(gold[word][:topn])),
        })
    return pd.DataFrame(rows), neighbours


def neighbour_overlap(kv_a, kv_b, words, topn=10):
    """How many of the top-n neighbours two embeddings share for each word."""
    rows = []
    for word in words:
        a, b = nearest_neighbors(kv_a, word, topn), nearest_neighbors(kv_b, word, topn)
        if a is None or b is None:
            rows.append({"word": word, "shared": None, "jaccard": None})
            continue
        set_a, set_b = {w for w, _ in a}, {w for w, _ in b}
        rows.append({"word": word, "shared": len(set_a & set_b),
                     "jaccard": len(set_a & set_b) / len(set_a | set_b)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 3D embedding space


def neighbourhood_points(kv, seeds, topn=25):
    """The seed words and their top-n neighbours, one row per word.

    Columns: word, group (the seed it belongs to, as typed), seed (bool), rank (0 for the seed),
    cosine. A word near several seeds stays in the first group; seeds missing from the vocabulary
    are skipped.
    """
    found = []
    for seed in seeds:
        key = lookup_word(kv, seed)
        if key is not None and key not in {k for _, k in found}:
            found.append((seed, key))
    rows = [{"word": key, "group": seed, "seed": True, "rank": 0, "cosine": 1.0} for seed, key in found]
    seen = {key for _, key in found}
    for seed, key in found:
        for rank, (word, cosine) in enumerate(kv.most_similar(key, topn=topn), start=1):
            if word not in seen:
                seen.add(word)
                rows.append({"word": word, "group": seed, "seed": False, "rank": rank, "cosine": float(cosine)})
    return pd.DataFrame(rows, columns=["word", "group", "seed", "rank", "cosine"])


def points_in(kv, points):
    """Reuse another embedding's neighbourhood_points() in kv.

    Keeps the rows whose word is in kv and recomputes ``cosine`` as the similarity to the group's
    seed in kv (NaN when the seed itself is missing); ``rank`` stays the rank in the original.
    """
    kept = points.assign(word=[lookup_word(kv, w) for w in points["word"]])
    kept = kept[kept["word"].notna()].reset_index(drop=True)
    seed_keys = {group: lookup_word(kv, group) for group in kept["group"].unique()}
    kept["cosine"] = [float(kv.similarity(word, seed_keys[group])) if seed_keys[group] else float("nan")
                      for word, group in zip(kept["word"], kept["group"])]
    return kept


def frequent_points(kv, n=1000, highlight=()):
    """The first n words of the vocabulary (usually the most frequent) plus any highlighted words."""
    words = list(kv.index_to_key[:n])
    marked = []
    for word in highlight:
        key = lookup_word(kv, word)
        if key is not None and key not in marked:
            marked.append(key)
    words += [w for w in marked if w not in set(words)]
    return pd.DataFrame({"word": words, "highlight": [w in set(marked) for w in words]})


def project_3d(vectors, method="pca", normalize=True, seed=42):
    """Project vectors to 3D. Returns (coords [n, 3], axis titles).

    With ``normalize`` the vectors are scaled to unit length first, so distances follow the cosine
    similarity used for neighbours. PCA axis titles carry the share of variance each axis keeps.
    """
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE

    X = np.asarray(vectors, dtype=np.float64)
    if X.ndim != 2 or len(X) < 4:
        raise ValueError("Need at least 4 words to draw a 3D projection (got %d)." % len(X))
    if normalize:
        X = X / np.maximum(np.linalg.norm(X, axis=1, keepdims=True), 1e-12)
    if method == "pca":
        pca = PCA(n_components=3, random_state=seed)
        coords = pca.fit_transform(X)
        titles = ["PC%d (%.0f%%)" % (i + 1, 100 * r) for i, r in enumerate(pca.explained_variance_ratio_)]
    elif method == "tsne":
        perplexity = float(min(30.0, (len(X) - 1) / 3.0))
        coords = TSNE(n_components=3, perplexity=perplexity, init="pca", random_state=seed,
                      metric="cosine" if not normalize else "euclidean").fit_transform(X)
        titles = ["t-SNE 1", "t-SNE 2", "t-SNE 3"]
    else:
        raise ValueError("Unknown projection %r" % method)
    return coords, titles


# --------------------------------------------------------------------------- personality regression (Table 3)


def _personality_data():
    from utils import dataset_utils
    raw = pd.read_csv(PERSONALITY_CSV, encoding="utf-8", sep=';')
    with contextlib.redirect_stdout(io.StringIO()):  # the helper prints a lot of debugging output
        documents, _, _, y = dataset_utils.process_personality_data(raw)
    return documents, np.asarray(y, dtype=float)


def _make_kv_model(name, kv, char_model):
    from word2vec.embedding_models import embedding_model
    model = embedding_model(name, kv.vector_size)
    model.model = kv
    model.set_char_model(char_model)
    return model


def build_personality_features(word_kv=None, private_kv=None, char_path=DEFAULT_CHAR_EMBEDDING, progress=None):
    """Same features as feature_extractors.text_features.Text_Features.feature_generator().

    Slots (in the paper's order): public word embedding (optional, the paper used GoogleNews),
    character embedding (required, unknown words fall back to it), private/DP embedding (optional).
    Returns X_train, X_test, y_train, y_test.
    """
    from feature_extractors import text_features
    from word2vec.embedding_models import embedding_model, embedding_models

    documents, y = _personality_data()
    extractor = text_features.Text_Features()
    documents = [extractor.normalise_fb_status(doc) for doc in documents]

    ngrams = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=1, lowercase=True,
                             max_features=1000, smooth_idf=True, norm='l2').fit_transform(documents).toarray()
    tfidf = TfidfVectorizer(ngram_range=(1, 3), min_df=1, lowercase=True,
                            max_features=500, smooth_idf=True, norm='l2').fit_transform(documents).toarray()
    features = np.append(ngrams, tfidf, 1)

    char_model = embedding_model("char2vec", 100)  # the real dimension is read from the file
    with contextlib.redirect_stdout(io.StringIO()):
        char_model.load_model(char_path)
    models = embedding_models([])
    if word_kv is not None:
        models.add_model(_make_kv_model("word2vec", word_kv, char_model), char_model)
    char_slot = embedding_model("char2vec", char_model.vector_dim)
    char_slot.model = char_model.model
    models.add_model(char_slot, char_model)
    if private_kv is not None:
        # KeyedVectors use the same lookup path as the "word2vec" slot (unknown words -> characters).
        models.add_model(_make_kv_model("word2vec", private_kv, char_model), char_model)

    embedding_features = []
    for i, text in enumerate(documents):
        tokens = extractor.tokenize_text(text, True)
        embedding_features.append([len(tokens), len(text)] + list(models.get_vector_of_document(tokens)))
        if progress and (i % 10 == 0 or i == len(documents) - 1):
            progress(i + 1, len(documents))
    features = np.append(features, np.asarray(embedding_features, dtype=float), 1)
    return train_test_split(features, y, test_size=0.2, random_state=42)


def fit_regressors(X_train, X_test, y_train, y_test, fit_svr=False):
    """RMSE of the paper's two regressors (same hyper-parameters as eval_personality_predictions.py)."""
    rmse = {}
    lr = linear_model.LinearRegression().fit(X_train, y_train)
    rmse["LR"] = float(mean_squared_error(y_test, lr.predict(X_test)) ** 0.5)
    if fit_svr:
        svr = svm.SVR(kernel='linear', C=1, epsilon=0.2).fit(X_train, y_train)
        rmse["SVR"] = float(mean_squared_error(y_test, svr.predict(X_test)) ** 0.5)
    return rmse


def reference_lr_table():
    """LR RMSE on the shipped cached features (the paper's own embeddings), in this environment."""
    rows = []

    def load(subdir):
        d = os.path.join(SAVED_FEATURES_DIR, subdir)
        return [np.load(os.path.join(d, n + ".npy")) for n in ("X_train_data", "X_test_data", "y_train_data", "y_test_data")]

    baseline = fit_regressors(*load("w2v_char2vec_no_private2v"))["LR"]
    for step in LEARNING_STEPS:
        rows.append({
            "learning_step": step,
            "Baseline (public only)": baseline,
            "+ DP embedding": fit_regressors(*load("saved_dp/w2v_%s_epochs" % step))["LR"],
            "+ non-DP embedding": fit_regressors(*load("saved_no_dp/w2v_%s_epochs" % step))["LR"],
        })
    return pd.DataFrame(rows)
