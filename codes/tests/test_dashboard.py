import functools
import gzip
import http.server
import os
import shutil
import tempfile
import threading
import unittest
import zipfile
from unittest import mock

import numpy as np
from gensim.models import KeyedVectors

from dashboard import embedding_sources as es
from dashboard import evaluations as ev
from tests.test_experiments import FIG2_CHAR_DP, FIG2_WORD_NODP

WORDS = ["time", "three", "been", "over", "not"]


def make_kv(n_extra=20, dim=8, seed=0):
    rng = np.random.default_rng(seed)
    words = WORDS + ["w%d" % i for i in range(n_extra)]
    kv = KeyedVectors(vector_size=dim)
    kv.add_vectors(words, rng.normal(size=(len(words), dim)).astype(np.float32))
    return kv


class SandboxTestCase(unittest.TestCase):
    """Redirect the dashboard's data/embeddings/ folders into a temp dir."""

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self._saved = {k: getattr(es, k) for k in ("EMBEDDINGS_DIR", "UPLOADS_DIR", "DOWNLOADS_DIR", "EXTRACTED_DIR",
                                                   "LIBRARY_DIRS")}
        es.EMBEDDINGS_DIR = os.path.join(self.tmp, "embeddings")
        es.UPLOADS_DIR = os.path.join(es.EMBEDDINGS_DIR, "uploads")
        es.DOWNLOADS_DIR = os.path.join(es.EMBEDDINGS_DIR, "downloads")
        es.EXTRACTED_DIR = os.path.join(es.EMBEDDINGS_DIR, "extracted")

    def tearDown(self):
        for k, v in self._saved.items():
            setattr(es, k, v)
        shutil.rmtree(self.tmp, ignore_errors=True)


class FormatTest(SandboxTestCase):
    def test_all_formats_round_trip(self):
        kv = make_kv()
        paths = {
            "word2vec_text": os.path.join(self.tmp, "a.txt"),
            "word2vec_binary": os.path.join(self.tmp, "a.bin"),
            "gensim_native": os.path.join(self.tmp, "a.kv"),
        }
        kv.save_word2vec_format(paths["word2vec_text"])
        kv.save_word2vec_format(paths["word2vec_binary"], binary=True)
        kv.save(paths["gensim_native"])
        glove = os.path.join(self.tmp, "glove.txt")
        with open(paths["word2vec_text"]) as src, open(glove, "w") as dst:
            dst.writelines(src.readlines()[1:])  # drop the header
        paths["text_no_header"] = glove
        gz = os.path.join(self.tmp, "a.txt.gz")
        with open(paths["word2vec_text"], "rb") as src, gzip.open(gz, "wb") as dst:
            shutil.copyfileobj(src, dst)

        for fmt, path in list(paths.items()) + [("word2vec_text", gz)]:
            with self.subTest(fmt=fmt, path=os.path.basename(path)):
                self.assertEqual(es.detect_format(path), fmt)
                loaded = es.load_keyed_vectors(path, allow_pickle=True)
                self.assertEqual(len(loaded), len(kv))
                np.testing.assert_allclose(loaded["time"], kv["time"], rtol=1e-5)
        self.assertEqual(len(es.load_keyed_vectors(paths["word2vec_text"], limit=3)), 3)

    def test_shipped_char_embeddings_skip_malformed_rows(self):
        kv = es.load_keyed_vectors(es.DEFAULT_CHAR_EMBEDDING)
        self.assertEqual(kv.vector_size, 300)
        self.assertEqual(len(kv), 89)  # 91 rows, 2 have a whitespace "word" and are skipped
        self.assertIn("a", kv.key_to_index)

    def test_pickle_formats_blocked_unless_trusted(self):
        path = os.path.join(self.tmp, "a.kv")
        make_kv().save(path)
        with self.assertRaises(es.EmbeddingSourceError):
            es.load_keyed_vectors(path)
        self.assertEqual(len(es.load_keyed_vectors(path, allow_pickle=True)), len(make_kv()))

    def test_zip_member_is_extracted_without_path_traversal(self):
        inner = os.path.join(self.tmp, "a.bin")
        make_kv().save_word2vec_format(inner, binary=True)
        archive = os.path.join(self.tmp, "vectors.zip")
        with zipfile.ZipFile(archive, "w") as zf:
            zf.write(inner, "deep/dir/a.bin")
            zf.writestr("../../escape.txt", "x 1 2 3\n")
        self.assertEqual(sorted(es.list_zip_members(archive)), ["../../escape.txt", "deep/dir/a.bin"])
        escaped = es.extract_zip_member(archive, "../../escape.txt")
        self.assertEqual(os.path.dirname(escaped), es.EXTRACTED_DIR)
        extracted = es.extract_zip_member(archive, "deep/dir/a.bin")
        self.assertEqual(len(es.load_keyed_vectors(extracted)), len(make_kv()))


class SourceTest(SandboxTestCase):
    def test_resolve_local_path_and_file_links(self):
        path = os.path.join(self.tmp, "my vectors.txt")
        make_kv().save_word2vec_format(path)
        self.assertEqual(es.resolve_local_path(path), path)
        self.assertEqual(es.resolve_local_path('"%s"' % path), path)  # quotes from "Copy as path"
        self.assertEqual(es.resolve_local_path(es.file_uri(path)), path)
        self.assertEqual(es.resolve_local_path("data/pretrained_models/public_embeddings/char-embeddings.txt"),
                         es.DEFAULT_CHAR_EMBEDDING)
        with self.assertRaises(es.EmbeddingSourceError):
            es.resolve_local_path(os.path.join(self.tmp, "missing.txt"))
        with self.assertRaises(es.EmbeddingSourceError):
            es.resolve_local_path("file://otherhost/etc/passwd")

    def test_normalize_url(self):
        self.assertEqual(es.normalize_url("https://www.dropbox.com/s/x/v.vec?dl=0"), "https://www.dropbox.com/s/x/v.vec?dl=1")
        self.assertEqual(es.normalize_url("https://github.com/o/r/blob/main/v.txt"),
                         "https://raw.githubusercontent.com/o/r/main/v.txt")
        self.assertEqual(es.normalize_url("https://huggingface.co/o/r/blob/main/v.txt"),
                         "https://huggingface.co/o/r/resolve/main/v.txt")
        for bad in ("ftp://host/v.txt", "javascript:alert(1)", "/no/scheme"):
            with self.assertRaises(es.EmbeddingSourceError):
                es.normalize_url(bad)

    def test_save_upload_sanitises_name(self):
        path = es.save_upload("../../evil name.txt", b"2 2\na 1 2\nb 3 4\n")
        self.assertEqual(os.path.dirname(path), es.UPLOADS_DIR)
        self.assertEqual(os.path.basename(path), "evil_name.txt")
        self.assertEqual(len(es.load_keyed_vectors(path)), 2)

    def test_download_from_local_http_server(self):
        serve_dir = os.path.join(self.tmp, "www")
        os.makedirs(serve_dir)
        make_kv().save_word2vec_format(os.path.join(serve_dir, "vectors.txt"))
        handler = functools.partial(http.server.SimpleHTTPRequestHandler, directory=serve_dir)
        handler.log_message = lambda *args: None
        server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        try:
            url = "http://127.0.0.1:%d/vectors.txt" % server.server_address[1]
            seen = []
            path = es.download(url, progress=lambda done, total: seen.append((done, total)))
            self.assertEqual(os.path.dirname(path), es.DOWNLOADS_DIR)
            self.assertEqual(seen[-1][0], seen[-1][1])
            self.assertEqual(len(es.load_keyed_vectors(path)), len(make_kv()))
            with self.assertRaises(es.EmbeddingSourceError):  # an HTML page is not an embedding file
                es.download("http://127.0.0.1:%d/" % server.server_address[1])
            with self.assertRaises(es.EmbeddingSourceError):
                es.download("http://127.0.0.1:%d/missing.txt" % server.server_address[1])
        finally:
            server.shutdown()
            server.server_close()


class EvaluationTest(unittest.TestCase):
    def test_reference_curves_match_figure2(self):
        ref = ev.paper_reference_curves()
        pick = lambda model, level: ref[(ref.model == model) & (ref.level == level)]["map"].tolist()  # noqa: E731
        np.testing.assert_allclose(pick("Non-DP (paper)", "word"), FIG2_WORD_NODP, rtol=1e-12)
        np.testing.assert_allclose(pick("DP (paper)", "character"), FIG2_CHAR_DP, rtol=1e-12)

    def test_semantic_scores_handle_missing_words(self):
        scores, neighbours = ev.semantic_space_scores(make_kv(n_extra=120))
        self.assertEqual(len(scores), len(ev.TESTED_WORDS))
        missing = scores[~scores["in_vocabulary"]]
        self.assertTrue((missing["map_word"] == 0).all())
        self.assertEqual(set(neighbours), set(WORDS) & set(ev.TESTED_WORDS))
        self.assertTrue(all(len(v) == 100 for v in neighbours.values()))

    def test_neighbour_overlap_with_itself(self):
        kv = make_kv()
        overlap = ev.neighbour_overlap(kv, kv, ["time", "missing"], topn=5)
        self.assertEqual(overlap.loc[0, "shared"], 5)
        self.assertEqual(overlap.loc[0, "jaccard"], 1.0)
        self.assertTrue(np.isnan(overlap.loc[1, "shared"]))


class Space3DTest(unittest.TestCase):
    def test_neighbourhoods_keep_seeds_and_drop_duplicates(self):
        kv = make_kv(n_extra=40)
        points = ev.neighbourhood_points(kv, ["Time", "three", "missing", "time"], topn=10)
        self.assertEqual(points.loc[points["seed"], "word"].tolist(), ["time", "three"])  # lowercase fallback
        self.assertEqual(points.loc[points["seed"], "group"].tolist(), ["Time", "three"])  # group = as typed
        self.assertFalse(points["word"].duplicated().any())
        self.assertNotIn("missing", set(points["group"]))
        # "three" stays its own seed even if it is one of "time"'s neighbours
        self.assertEqual(points[points["word"] == "three"]["group"].tolist(), ["three"])

    def test_points_in_other_embedding_recomputes_cosine(self):
        kv = make_kv(n_extra=40)
        points = ev.neighbourhood_points(kv, ["time", "three"], topn=10)
        other = make_kv(n_extra=5, seed=1)  # smaller vocabulary, different vectors
        moved = ev.points_in(other, points)
        self.assertTrue(set(moved["word"]) <= set(other.key_to_index))
        row = moved[moved["word"] == "w0"].iloc[0]
        self.assertAlmostEqual(row["cosine"], float(other.similarity("w0", row["group"])), places=5)

    def test_frequent_points_add_highlights(self):
        kv = make_kv(n_extra=40)
        points = ev.frequent_points(kv, n=10, highlight=["w30", "time", "nope"])
        self.assertEqual(len(points), 11)  # top 10 + w30
        self.assertEqual(points.loc[points["highlight"], "word"].tolist(), ["time", "w30"])

    def test_projections(self):
        kv = make_kv(n_extra=60)
        vectors = kv[kv.index_to_key]
        coords, titles = ev.project_3d(vectors, "pca")
        self.assertEqual(coords.shape, (len(kv), 3))
        self.assertTrue(titles[0].startswith("PC1 ("))
        coords, titles = ev.project_3d(vectors, "tsne", normalize=False)
        self.assertEqual(coords.shape, (len(kv), 3))
        with self.assertRaises(ValueError):
            ev.project_3d(vectors[:3], "pca")


class SampleTest(SandboxTestCase):
    def test_samples_skip_hand_added_files_and_zips(self):
        es.LIBRARY_DIRS = [es.EMBEDDINGS_DIR]
        for path in ["mine.txt", "uploads/up.txt", "downloads/web.txt", "extracted/x__a.txt", "bundle.zip"]:
            full = os.path.join(es.EMBEDDINGS_DIR, path)
            os.makedirs(os.path.dirname(full), exist_ok=True)
            make_kv().save_word2vec_format(full)
        self.assertEqual([os.path.basename(e.path) for e in es.list_samples()], ["mine.txt"])
        self.assertEqual(len(es.list_library()), 5)

    def test_build_text8_sample_quickly(self):
        from dashboard import make_sample_embeddings
        path, n_words, _ = make_sample_embeddings.build_text8_sample(epochs=1, max_tokens=200_000, workers=2)
        self.assertEqual(path, es.text8_sample_path())
        kv = es.load_keyed_vectors(path)
        self.assertEqual((len(kv), kv.vector_size), (n_words, 200))
        self.assertIn("three", kv.key_to_index)


try:
    from streamlit.testing.v1 import AppTest
except ImportError:  # the dashboard extras are optional
    AppTest = None


@unittest.skipIf(AppTest is None, "streamlit is not installed")
class DashboardAppTest(SandboxTestCase):
    APP = os.path.join(es.CODES_DIR, "dashboard", "app.py")

    def setUp(self):
        super().setUp()
        # Most tests start empty; test_samples_load_on_start turns auto-loading back on.
        env = mock.patch.dict(os.environ, {"DPUGC_AUTOLOAD_SAMPLES": "0"})
        env.start()
        self.addCleanup(env.stop)

    def test_samples_load_on_start_and_render_in_3d(self):
        os.environ["DPUGC_AUTOLOAD_SAMPLES"] = "1"
        os.makedirs(es.EMBEDDINGS_DIR)
        make_kv(n_extra=200).save_word2vec_format(os.path.join(es.EMBEDDINGS_DIR, "sample.txt"))
        es.LIBRARY_DIRS = [es.EMBEDDINGS_DIR, os.path.dirname(es.DEFAULT_CHAR_EMBEDDING)]
        at = AppTest.from_file(self.APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        loaded = sorted((e["name"], e["source"]) for e in at.session_state["embeddings"].values())
        self.assertEqual(loaded, [("char-embeddings", "project sample"), ("sample", "project sample")])
        self.assertTrue(any("2 project samples" in s.value for s in at.success))
        self.assertEqual(len(at.get("plotly_chart")), 1)  # neighbourhoods of the default words in "sample"
        # Removed samples stay removed on later reruns; "Load all samples" brings them back.
        sample_id = next(e["id"] for e in at.session_state["embeddings"].values() if e["name"] == "sample")
        at.button(key="rm_" + sample_id).click().run()
        self.assertEqual(len(at.session_state["embeddings"]), 1)
        at.button(key="load_samples").click().run()
        self.assertEqual(len(at.session_state["embeddings"]), 2)

    def test_3d_modes_and_side_by_side(self):
        path = os.path.join(self.tmp, "a.txt")
        make_kv(n_extra=200).save_word2vec_format(path)
        at = AppTest.from_file(self.APP, default_timeout=60).run()
        for name in ("a", "b"):
            at.text_input(key="new_name").input(name)
            at.text_input(key="link_input").input(path)
            at.button(key="load_link").click().run()
        at.selectbox(key="space_cmp").set_value("b").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("plotly_chart")), 2)
        at.radio(key="space_method").set_value("t-SNE").run()
        at.radio(key="space_mode").set_value("Most frequent words").run()
        at.text_input(key="space_highlight").input("time, nope").run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.get("plotly_chart")), 2)
        self.assertTrue(any("nope" in c.value for c in at.caption))

    def test_load_char_embeddings_by_path_and_explore(self):
        at = AppTest.from_file(self.APP, default_timeout=60).run()
        self.assertFalse(at.exception)
        at.text_input(key="link_input").input("data/pretrained_models/public_embeddings/char-embeddings.txt")
        at.button(key="load_link").click().run()
        self.assertFalse(at.exception)
        self.assertEqual(len(at.session_state["embeddings"]), 1)
        entry = next(iter(at.session_state["embeddings"].values()))
        self.assertEqual((entry["words"], entry["dim"]), (89, 300))
        at.text_input(key="explore_words").input("a, e").run()
        self.assertFalse(at.exception)
        self.assertFalse(at.error)

    def test_two_embeddings_are_both_scored_and_removable(self):
        at = AppTest.from_file(self.APP, default_timeout=60).run()
        for name in ("first", "second"):
            at.text_input(key="new_name").input(name)
            at.text_input(key="link_input").input("data/pretrained_models/public_embeddings/char-embeddings.txt")
            at.button(key="load_link").click().run()
            self.assertFalse(at.exception)
        self.assertEqual(at.multiselect(key="semantic_sel").value, ["first", "second"])
        self.assertFalse(at.warning)
        first_id = next(e["id"] for e in at.session_state["embeddings"].values() if e["name"] == "first")
        at.button(key="rm_" + first_id).click().run()
        self.assertFalse(at.exception)
        self.assertEqual(at.multiselect(key="semantic_sel").value, ["second"])

    def test_bad_path_shows_error(self):
        at = AppTest.from_file(self.APP, default_timeout=60).run()
        at.text_input(key="link_input").input("/definitely/not/here.txt")
        at.button(key="load_link").click().run()
        self.assertFalse(at.exception)
        self.assertTrue(any("Not found" in e.value for e in at.error))


if __name__ == "__main__":
    unittest.main()
