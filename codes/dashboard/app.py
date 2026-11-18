"""dpUGC embedding dashboard.

Load word embeddings from the project, a local path / file:// link, an upload or a web link, then
explore them and evaluate them with the paper's two experiments.

Run from the repository root:  ./codes/run_dashboard.sh   (opens http://localhost:8501)
"""
import os
import sys
import time
import uuid

CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if CODES_DIR not in sys.path:  # `streamlit run` only puts dashboard/ on the path
    sys.path.insert(0, CODES_DIR)

import altair as alt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import plotly.graph_objects as go  # noqa: E402
import streamlit as st  # noqa: E402


@st.cache_resource(show_spinner=False)
def _project_module_mtimes():
    return {}  # module name -> mtime of its file when this server process imported it


def _project_modules():
    """{name: file} of the loaded modules that live in codes/ (dashboard, utils, word2vec, ...)."""
    found = {}
    for name, module in list(sys.modules.items()):
        path = getattr(module, "__file__", None)
        if name != "__main__" and path and os.path.abspath(path).startswith(CODES_DIR + os.sep):
            found[name] = path
    return found


def _mtime(path):
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


def refresh_project_modules():
    """Re-import the project modules if any of them changed on disk since it was imported.

    Streamlit re-reads this file on every run but keeps imported modules in memory, so after a code
    update a running server would mix the new app.py with old helper modules ("module ... has no
    attribute ..."). Streamlit's file watcher only fixes that for the browser session that saw the
    change, not after a page reload, so check the files here, on every run.
    """
    seen = _project_module_mtimes()
    modules = _project_modules()
    if any(name in seen and seen[name] != _mtime(path) for name, path in modules.items()):
        for name in modules:
            sys.modules.pop(name, None)
        seen.clear()


def record_project_modules():
    seen = _project_module_mtimes()
    for name, path in _project_modules().items():
        seen.setdefault(name, _mtime(path))


refresh_project_modules()
from dashboard import embedding_sources as es  # noqa: E402
from dashboard import evaluations as ev  # noqa: E402
from dashboard import make_sample_embeddings  # noqa: E402
record_project_modules()

st.set_page_config(page_title="dpUGC Embedding Dashboard", page_icon=":material/hub:", layout="wide")

# Validated categorical palette (fixed order; slots 0-1 = the paper's DP / non-DP models).
SERIES = {
    "light": ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"],
    "dark": ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#0ca30c", "#9085e9", "#e66767"],
}
USER_SLOTS = list(range(2, 8))  # at most 6 user embeddings get a chart colour
STEP_LABELS = {"20": "20", "200": "200", "500": "500", "1000": "1K", "5000": "5K", "10000": "10K",
               "50000": "50K", "90000": "90K", "100000": "100K"}
UPLOAD_TYPES = ["txt", "vec", "bin", "gz", "zip", "w2v", "emb"]
# The project samples (see es.list_samples) load when a session starts; DPUGC_AUTOLOAD_SAMPLES=0 turns it off.
AUTOLOAD_SAMPLES = os.environ.get("DPUGC_AUTOLOAD_SAMPLES", "1") != "0"
SAMPLE_WORD_LIMIT = 100_000  # keeps start-up quick if a very large file was copied into data/embeddings/
SPACE_SEEDS = "time, three, government, good, music, france"
MAX_SEEDS = 8  # one categorical colour each

ss = st.session_state
ss.setdefault("embeddings", {})      # id -> entry dict (includes the loaded KeyedVectors under "kv")
ss.setdefault("pending_zip", None)   # (zip_path, trusted, name, limit) waiting for a member choice
ss.setdefault("regression_runs", [])
ss.setdefault("feature_cache", {})
ss.setdefault("flash", None)
ss.setdefault("downloads", {})       # url -> downloaded local path


def palette():
    theme = getattr(getattr(st.context, "theme", None), "type", None)
    return SERIES["dark" if theme == "dark" else "light"]


# --------------------------------------------------------------------------- cached work


@st.cache_resource(show_spinner=False, max_entries=8)
def _load_cached(path, fmt, limit, mtime, size, allow_pickle):
    # mtime/size are part of the cache key so an edited file is re-read.
    return es.load_keyed_vectors(path, fmt=fmt, limit=limit or None, allow_pickle=allow_pickle)


@st.cache_data(show_spinner=False)
def reference_curves():
    return ev.paper_reference_curves()


@st.cache_data(show_spinner="Fitting linear regression on the shipped features...")
def reference_lr():
    return ev.reference_lr_table()


def _stat(path):
    if os.path.isdir(path):
        files = [os.path.join(path, f) for f in ("final_embeddings.pkl", "words_dictionary.pkl")]
        return max(os.path.getmtime(f) for f in files), sum(os.path.getsize(f) for f in files)
    return os.path.getmtime(path), os.path.getsize(path)


def register(path, source, trusted, name, limit):
    """Load ``path`` and add it to the session's embeddings. Returns the entry or None (error shown)."""
    fmt = es.detect_format(path)
    if fmt == "zip":
        members = [m for m in es.list_zip_members(path) if m.lower().endswith(es.EMBEDDING_SUFFIXES)]
        if len(members) == 1:
            path = es.extract_zip_member(path, members[0])
            fmt = es.detect_format(path)
        else:
            ss.pending_zip = (path, trusted, name, limit)
            return None
    if fmt in es.PICKLE_FORMATS and not trusted:
        raise es.EmbeddingSourceError("%s from an upload or web link is not loaded: pickle files can run code. "
                                      "Save it locally and use its path instead." % es.FORMAT_LABELS[fmt])
    mtime, size = _stat(path)
    started = time.time()
    with st.spinner("Loading %s ..." % os.path.basename(path.rstrip("/"))):
        kv = _load_cached(path, fmt, int(limit or 0), mtime, size, trusted)
    used = {e["slot"] for e in ss.embeddings.values()}
    slot = next((s for s in USER_SLOTS if s not in used), None)
    entry_id = uuid.uuid4().hex[:8]
    entry = {
        "id": entry_id, "name": unique_name(name or default_name(path)), "path": path, "format": fmt,
        # Only report a word limit that actually cut the file short.
        "source": source, "limit": int(limit) if limit and len(kv) >= int(limit) else 0, "slot": slot, "kv": kv,
        "words": len(kv), "dim": kv.vector_size, "seconds": time.time() - started,
    }
    ss.embeddings[entry_id] = entry
    # Add it to the Figure 2 selection too (the widget keeps its old value once it exists).
    if slot is not None and "semantic_sel" in ss and len(ss["semantic_sel"]) < 6:
        ss["semantic_sel"] = list(ss["semantic_sel"]) + [entry["name"]]
    ss.flash = "Loaded **%s**: %s words x %s dimensions." % (entry["name"], f"{entry['words']:,}", entry["dim"])
    return entry


def default_name(path):
    path = path.rstrip("/")
    base = os.path.basename(path)
    if es.is_dpugc_dir(path):  # e.g. trained_models_dp_dpsgd/_20epoch -> trained_models_dp_dpsgd@20epoch
        return "%s@%s" % (os.path.basename(os.path.dirname(path)), base.lstrip("_"))
    for ext in (".gz", ".txt", ".vec", ".bin", ".w2v", ".emb", ".kv", ".model"):
        if base.lower().endswith(ext):
            base = base[: -len(ext)]
    return base or "embedding"


NONE_OPTION = "None"
SHIPPED_CHAR_OPTION = "Shipped char-embeddings.txt"


def unique_name(name):
    names = {e["name"] for e in ss.embeddings.values()} | {NONE_OPTION, SHIPPED_CHAR_OPTION}
    candidate, n = name, 2
    while candidate in names:
        candidate, n = "%s (%d)" % (name, n), n + 1
    return candidate


def run_safely(fn, *args):
    try:
        return fn(*args)
    except es.EmbeddingSourceError as ex:
        st.error(str(ex))
    except Exception as ex:  # surface unexpected loader errors instead of a stack trace
        st.error("Could not load the embeddings: %s: %s" % (type(ex).__name__, ex))
    return None


def load_samples():
    """Load the project samples that are not loaded yet (while chart colours are free)."""
    loaded_paths = {e["path"] for e in ss.embeddings.values()}
    loaded = []
    for sample in es.list_samples():
        if sample.path in loaded_paths:
            continue
        if {e["slot"] for e in ss.embeddings.values()} >= set(USER_SLOTS):
            break  # every chart colour is taken
        item = run_safely(register, sample.path, "project sample", True, None, SAMPLE_WORD_LIMIT)
        if item:
            loaded.append(item)
    if loaded:
        ss.flash = "Loaded %d project sample%s: %s." % (
            len(loaded), "" if len(loaded) == 1 else "s",
            ", ".join("**%s** (%s words x %s dims)" % (e["name"], f"{e['words']:,}", e["dim"]) for e in loaded))
    return loaded


# --------------------------------------------------------------------------- header


st.title("dpUGC embedding dashboard")
st.caption("Load word embeddings, explore their neighbours, and evaluate them with the two experiments of "
           "*dpUGC: Learn Differentially Private Representation for User Generated Contents* (CICLing 2019).")

if AUTOLOAD_SAMPLES and not ss.get("samples_autoloaded"):
    ss.samples_autoloaded = True  # once per browser session; removed samples stay removed
    load_samples()

# --------------------------------------------------------------------------- 1. add embeddings

st.header("1. Add embeddings", divider="gray")
opt_cols = st.columns([2, 1])
with opt_cols[0]:
    new_name = st.text_input("Display name (optional)", placeholder="defaults to the file name",
                             key="new_name")
with opt_cols[1]:
    limit = st.number_input("Max words to load (0 = all)", min_value=0, value=0, step=50000,
                            help="Embedding files are usually sorted by frequency, so a limit keeps the most "
                                 "frequent words. Useful for very large files such as GoogleNews (3M words).")

# Stable keys keep the selected tab when elements above it come and go between reruns.
tab_link, tab_upload, tab_library = st.tabs([":material/link: Paste a path or link",
                                             ":material/upload: Upload a file",
                                             ":material/folder_open: Project library"], key="source_tabs")

with tab_link:
    st.markdown("Paste **a local file path** (absolute, `~/...`, or relative to the repository), a "
                "**`file://` link**, or an **`http(s)://` direct download link**. Local files are loaded "
                "in place, without copying. Web files are downloaded to `data/embeddings/downloads/`.")
    link = st.text_input("Path or link", key="link_input", label_visibility="collapsed",
                         placeholder="/path/to/vectors.txt   ·   file:///path/to/model.bin   ·   "
                                     "https://example.org/vectors.vec.gz")
    # Not disabled while the box is empty: a typed value only commits when the box loses focus, so a
    # disabled button would swallow the first click.
    if st.button("Load", type="primary", key="load_link", icon=":material/download:"):
        text = link.strip().strip('"').strip("'")
        if not text:
            st.warning("Paste a file path or link first.")
        elif text.lower().startswith(("http://", "https://")):
            bar = st.progress(0.0, text="Connecting...")
            last = [0.0]

            def report(done, total):
                if time.time() - last[0] < 0.25 and (total is None or done < total):
                    return
                last[0] = time.time()
                if total:
                    bar.progress(min(done / total, 1.0), text="Downloading %s of %s" % (es.human_size(done),
                                                                                        es.human_size(total)))
                else:
                    bar.progress(0.0, text="Downloading %s" % es.human_size(done))

            local = ss.downloads.get(text)
            if not (local and os.path.exists(local)):  # don't download the same link twice
                local = run_safely(es.download, text, report)
                if local:
                    ss.downloads[text] = local
            bar.empty()
            if local:
                run_safely(register, local, "web link", False, new_name, limit)
        else:
            local = run_safely(es.resolve_local_path, text)
            if local:
                run_safely(register, local, "local path", True, new_name, limit)

with tab_upload:
    st.markdown("Upload an embedding file from your computer (up to 1 GB). The file is saved to "
                "`data/embeddings/uploads/`, so it shows up in the project library next time. For "
                "bigger files, paste the file's path instead.")
    uploaded = st.file_uploader("Embedding file", type=UPLOAD_TYPES, label_visibility="collapsed")
    if st.button("Load upload", type="primary", key="load_upload", icon=":material/upload:"):
        if uploaded is None:
            st.warning("Choose a file first.")
        else:
            path = es.save_upload(uploaded.name, uploaded.getvalue())
            run_safely(register, path, "upload", False, new_name or default_name(uploaded.name), limit)

with tab_library:
    entries = es.list_library()
    st.markdown("Files in `data/embeddings/` (copy files there to add them), the shipped public embeddings, "
                "and models trained with `codes/10.run_train_dp_embedding.sh` (`data/text8_data/`). "
                "Everything here except uploads, downloads and zip archives is a **project sample**: samples "
                "load automatically when the dashboard opens.")
    if entries:
        by_label = {"%s  (%s)" % (e.label, es.human_size(e.size_bytes)): e for e in entries}
        choice = st.selectbox("Embedding", options=list(by_label), label_visibility="collapsed", key="library_choice")
        lib_cols = st.columns([1, 1, 1.5, 3.5])
        if lib_cols[0].button("Load", type="primary", key="load_library", icon=":material/download:"):
            run_safely(register, by_label[choice].path, "project library", True, new_name, limit)
        lib_cols[1].button("Refresh", key="refresh_library", icon=":material/refresh:")
        if lib_cols[2].button("Load all samples", key="load_samples", icon=":material/library_add:"):
            if not load_samples():
                st.info("All project samples are already loaded (or every chart colour is taken).")
    else:
        st.info("No embedding files found yet.")
    if not os.path.exists(es.text8_sample_path()):
        with st.container(border=True):
            st.markdown("**No word-level sample yet.** Build one from `data/text8.zip`: a gensim skip-gram "
                        "word2vec (200 dimensions, 50,000 words, no DP) saved as `data/embeddings/%s`. "
                        "Takes about a minute; from a terminal: `cd codes && python -m "
                        "dashboard.make_sample_embeddings`." % es.TEXT8_SAMPLE_NAME)
            if st.button("Build the text8 sample", key="build_sample", icon=":material/model_training:"):
                with st.spinner("Training word2vec on text8 (about a minute)..."):
                    built = run_safely(make_sample_embeddings.build_text8_sample)
                if built:
                    run_safely(register, built[0], "project sample", True, None, SAMPLE_WORD_LIMIT)
                    st.rerun()

if ss.pending_zip:
    zip_path, trusted, pending_name, pending_limit = ss.pending_zip
    members = es.list_zip_members(zip_path)
    with st.container(border=True):
        st.markdown("**%s** contains several files. Pick the embedding file to load:" % os.path.basename(zip_path))
        member = st.selectbox("File in archive", members, key="zip_member")
        zip_cols = st.columns([1, 1, 6])
        if zip_cols[0].button("Extract and load", type="primary", key="zip_load"):
            ss.pending_zip = None
            extracted = run_safely(es.extract_zip_member, zip_path, member)
            if extracted:
                run_safely(register, extracted, "zip", trusted, pending_name, pending_limit)
            st.rerun()
        if zip_cols[1].button("Cancel", key="zip_cancel"):
            ss.pending_zip = None
            st.rerun()

# Shown here (below the load buttons) so it appears in the same run that loaded the embeddings. The
# placeholder always exists, so the layout keeps the same elements whether or not a message shows.
flash_slot = st.empty()
if ss.flash:
    flash_slot.success(ss.flash, icon=":material/check_circle:")
    ss.flash = None

# --------------------------------------------------------------------------- loaded embeddings

st.subheader("Loaded embeddings")
if not ss.embeddings:
    st.info("Nothing loaded. The project samples load when the dashboard opens; to get them back, use "
            "**Project library > Load all samples**.", icon=":material/info:")
else:
    colors = palette()
    for loaded in list(ss.embeddings.values()):
        with st.container(border=True):
            cols = st.columns([0.3, 3, 2.2, 1.6, 1.4])
            swatch = colors[loaded["slot"]] if loaded["slot"] is not None else "transparent"
            cols[0].markdown("<div style='width:14px;height:14px;border-radius:4px;margin-top:6px;"
                             "background:%s'></div>" % swatch, unsafe_allow_html=True)
            cols[1].markdown("**%s**  \n%s words · %s dims · %s" % (
                loaded["name"], f"{loaded['words']:,}", loaded["dim"], es.FORMAT_LABELS.get(loaded["format"], loaded["format"])))
            cols[2].markdown("Source: %s%s  \nLoaded in %.1f s" % (
                loaded["source"], " (first %s words)" % f"{loaded['limit']:,}" if loaded["limit"] else "", loaded["seconds"]))
            with cols[3].popover("Path / link", icon=":material/link:"):
                st.caption("Paste this link into *Paste a path or link* to load the file again.")
                st.code(es.file_uri(loaded["path"]), language=None, wrap_lines=True)
            if cols[4].button("Remove", key="rm_" + loaded["id"], icon=":material/delete:", width="stretch"):
                del ss.embeddings[loaded["id"]]
                ss.feature_cache = {k: v for k, v in ss.feature_cache.items() if loaded["id"] not in k}
                st.rerun()


def embedding_options():
    """Names of the loaded embeddings (unique), used as selector values."""
    return [e["name"] for e in ss.embeddings.values()]


def entry(name):
    return next(e for e in ss.embeddings.values() if e["name"] == name)


# --------------------------------------------------------------------------- 2. evaluate

st.header("2. Explore and evaluate", divider="gray")
tab_explore, tab_space, tab_semantic, tab_regression = st.tabs([
    ":material/travel_explore: Explore neighbours",
    ":material/3d_rotation: 3D embedding space",
    ":material/show_chart: Semantic space (Figure 2)",
    ":material/functions: Personality regression (Table 3)",
], key="eval_tabs")

with tab_explore:
    if not ss.embeddings:
        st.info("Load an embedding first.")
    else:
        ex_cols = st.columns([2, 3, 1])
        primary = ex_cols[0].selectbox("Embedding", embedding_options(), key="explore_emb")
        words_text = ex_cols[1].text_input("Words (comma separated)", value="time, three, government, good",
                                           key="explore_words")
        topn = ex_cols[2].number_input("Neighbours", 1, 100, 10, key="explore_topn")
        kv = entry(primary)["kv"]
        words = [w.strip() for w in words_text.split(",") if w.strip()][:8]
        if words:
            word_cols = st.columns(min(len(words), 4))
            for i, word in enumerate(words):
                with word_cols[i % len(word_cols)]:
                    neighbours = ev.nearest_neighbors(kv, word, int(topn))
                    st.markdown("**%s**" % word)
                    if neighbours is None:
                        st.caption("Not in the vocabulary.")
                    else:
                        st.dataframe(pd.DataFrame(neighbours, columns=["neighbour", "cosine"]),
                                     hide_index=True, column_config={"cosine": st.column_config.NumberColumn(format="%.3f")})

        st.markdown("##### Similarity of two words")
        sim_cols = st.columns([2, 2, 3])
        w1 = sim_cols[0].text_input("Word A", value="good", key="sim_a")
        w2 = sim_cols[1].text_input("Word B", value="bad", key="sim_b")
        k1, k2 = ev.lookup_word(kv, w1.strip()), ev.lookup_word(kv, w2.strip())
        if k1 and k2:
            sim_cols[2].metric("Cosine similarity", "%.3f" % kv.similarity(k1, k2))
        else:
            sim_cols[2].caption("Both words must be in the vocabulary.")

        if len(ss.embeddings) > 1:
            st.markdown("##### Compare two embeddings")
            st.caption("How many of the top-k neighbours are shared, e.g. a DP model versus its non-DP twin.")
            cmp_cols = st.columns([2, 2, 1])
            others = [e for e in embedding_options() if e != primary]
            other = cmp_cols[0].selectbox("Compare with", others, key="cmp_other")
            cmp_k = cmp_cols[2].number_input("Top-k", 1, 100, 10, key="cmp_k")
            cmp_words = list(dict.fromkeys(words + ev.TESTED_WORDS))
            overlap = ev.neighbour_overlap(kv, entry(other)["kv"], cmp_words, int(cmp_k))
            cmp_cols[1].metric("Mean shared neighbours", "-" if overlap["shared"].isna().all()
                               else "%.1f of %d" % (overlap["shared"].mean(), cmp_k))
            st.dataframe(overlap, hide_index=True,
                         column_config={"jaccard": st.column_config.ProgressColumn("Jaccard", min_value=0.0,
                                                                                    max_value=1.0, format="%.2f")})


@st.cache_data(show_spinner=False, max_entries=32)
def projection(entry_id, words, method, normalize, _kv):
    """3D coordinates of ``words`` in one loaded embedding (entry_id + words identify the cached result)."""
    return ev.project_3d(_kv[list(words)], method=method, normalize=normalize)


# 3D pane / grid colours per theme. Opaque on purpose: plotly's WebGL scene ignores alpha here.
SCENE_COLORS = {"light": ("#f4f5f7", "#dcdee3"), "dark": ("#191c23", "#343844")}


def _axis(title):
    # Recessive panes and grid give depth cues without competing with the points.
    pane, grid = SCENE_COLORS["dark" if palette() is SERIES["dark"] else "light"]
    return dict(title=dict(text=title, font=dict(size=12)), showticklabels=False, showbackground=True,
                backgroundcolor=pane, gridcolor=grid, zeroline=False, showspikes=False)


def space_figure(points, coords, titles, color_of, label_all, uirevision):
    """3D scatter: one trace (and legend entry) per colour group; seeds / highlights are labelled."""
    df = points.assign(x=coords[:, 0], y=coords[:, 1], z=coords[:, 2])
    fig = go.Figure()
    if "group" in df:  # neighbourhoods: group = seed word
        for group, part in df.groupby("group", sort=False):
            hover = ["<b>%s</b><br>%s" % (w, "chosen word" if s else
                                         "near <i>%s</i> (cosine %s)" % (group, "n/a" if np.isnan(c) else "%.3f" % c))
                     for w, s, c in zip(part["word"], part["seed"], part["cosine"])]
            fig.add_trace(go.Scatter3d(
                x=part["x"], y=part["y"], z=part["z"], mode="markers+text",
                name="%s (%d)" % (group, int((~part["seed"]).sum())),
                text=[w if s or label_all else "" for w, s in zip(part["word"], part["seed"])],
                textposition="top center", hovertext=hover, hovertemplate="%{hovertext}<extra></extra>",
                marker=dict(size=[10 if s else 5 for s in part["seed"]], color=color_of[group], opacity=0.9,
                            symbol=["diamond" if s else "circle" for s in part["seed"]], line=dict(width=0))))
    else:  # most frequent words, with optional highlighted words
        for marked, part in df.groupby("highlight", sort=True):
            fig.add_trace(go.Scatter3d(
                x=part["x"], y=part["y"], z=part["z"], mode="markers+text" if marked or label_all else "markers",
                name="Highlighted words (%d)" % len(part) if marked else "Frequent words",
                text=part["word"] if marked or label_all else None, textposition="top center",
                hovertext=part["word"], hovertemplate="<b>%{hovertext}</b><extra></extra>",
                marker=dict(size=9 if marked else 3.5, color=color_of[marked], opacity=1.0 if marked else 0.7,
                            symbol="diamond" if marked else "circle", line=dict(width=0))))
    fig.update_layout(
        height=620, margin=dict(l=0, r=0, t=10, b=0), uirevision=uirevision, showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, itemsizing="constant"),
        scene=dict(xaxis=_axis(titles[0]), yaxis=_axis(titles[1]), zaxis=_axis(titles[2]), aspectmode="cube",
                   camera=dict(eye=dict(x=1.05, y=1.05, z=0.7))))  # closer than plotly's default view
    fig.update_traces(textfont=dict(size=13))
    return fig


with tab_space:
    if not ss.embeddings:
        st.info("Load an embedding first.")
    else:
        st.markdown("Projects word vectors to 3D so you can see which words cluster together. Drag to rotate and "
                    "hover a point to see its word. To zoom, pick the magnifier in the chart's toolbar and drag up or "
                    "down; the house icon resets the view. Click a legend entry to hide a group.")
        names = embedding_options()
        sp_cols = st.columns([2, 2, 1.3])
        sp_name = sp_cols[0].selectbox("Embedding", names, key="space_emb")
        sp_other = sp_cols[1].selectbox("Compare side by side with", [NONE_OPTION] + [n for n in names if n != sp_name],
                                        key="space_cmp")
        method = sp_cols[2].radio("Projection", ["PCA", "t-SNE"], horizontal=True, key="space_method",
                                  help="PCA is a linear view of the largest directions of variation (instant). "
                                       "t-SNE keeps close words close and separates clusters, but distances "
                                       "between clusters mean little (a few seconds).")
        mode = st.radio("Words to plot", ["Neighbourhoods of chosen words", "Most frequent words"], horizontal=True,
                        key="space_mode")
        neighbourhoods = mode.startswith("Neighbourhoods")
        if neighbourhoods:
            w_cols = st.columns([4, 1])
            seeds_text = w_cols[0].text_input("Words (comma separated, up to %d, one colour each)" % MAX_SEEDS,
                                              value=SPACE_SEEDS, key="space_seeds")
            sp_topn = int(w_cols[1].number_input("Neighbours per word", 5, 100, 25, step=5, key="space_topn"))
            seeds = list(dict.fromkeys(w.strip() for w in seeds_text.split(",") if w.strip()))
            if len(seeds) > MAX_SEEDS:
                st.caption("Only the first %d words are plotted." % MAX_SEEDS)
                seeds = seeds[:MAX_SEEDS]
            colors = palette()
            color_of = {seed: colors[i] for i, seed in enumerate(seeds)}  # fixed by position, never by rank
        else:
            w_cols = st.columns([1, 4])
            sp_n = int(w_cols[0].number_input("Number of words", 50, 5000, 1000, step=250, key="space_n",
                                              help="Taken from the top of the vocabulary, which word2vec files "
                                                   "sort by frequency."))
            highlight_text = w_cols[1].text_input("Highlight words (comma separated)", value="", key="space_highlight",
                                                  placeholder="e.g. king, queen, france, paris")
            highlights = [w.strip() for w in highlight_text.split(",") if w.strip()]
            colors = palette()
            color_of = {False: colors[0], True: colors[1]}
        o_cols = st.columns(3)
        label_all = o_cols[0].toggle("Label every point", key="space_labels")
        normalize = o_cols[1].toggle("Unit-length vectors (cosine)", value=True, key="space_norm",
                                     help="Scale vectors to length 1 first, so the layout follows cosine similarity, "
                                          "the measure used for neighbours.")
        reuse = o_cols[2].toggle("Same words in both", value=True, key="space_reuse",
                                 disabled=sp_other == NONE_OPTION or not neighbourhoods,
                                 help="Plot the first embedding's neighbourhoods in the second one too, coloured the "
                                      "same way: if the colours stay together, the second space keeps that structure "
                                      "(e.g. a DP model versus its non-DP twin).")

        shown = [sp_name] + ([] if sp_other == NONE_OPTION else [sp_other])
        chart_cols = st.columns(len(shown))
        first_points = None
        tables = []
        for i, name in enumerate(shown):
            item = entry(name)
            kv = item["kv"]
            with chart_cols[i]:
                if len(shown) > 1:
                    st.markdown("**%s**" % name)
                if neighbourhoods:
                    if i == 1 and reuse and first_points is not None:
                        points = ev.points_in(kv, first_points)
                    else:
                        points = ev.neighbourhood_points(kv, seeds, sp_topn)
                    if i == 0:
                        first_points = points
                    missing = [s for s in seeds if s not in set(points["group"])]
                else:
                    points = ev.frequent_points(kv, sp_n, highlights)
                    missing = [w for w in highlights if ev.lookup_word(kv, w) is None]
                if len(points) < 4:
                    st.info("Need at least 4 words in this vocabulary to draw a 3D projection. Try other words or "
                            "*Most frequent words*.%s" % (" Not in this vocabulary: %s." % ", ".join(missing)
                                                          if missing else ""))
                    continue
                with st.spinner("Projecting %s words with %s..." % (f"{len(points):,}", method)):
                    coords, titles = projection(item["id"], tuple(points["word"]), method.lower().replace("-", ""),
                                                normalize, kv)
                fig = space_figure(points, coords, titles, color_of, label_all,
                                   uirevision="%s|%s|%s" % (item["id"], method, mode))
                # No wheel zoom: scrolling the page over the chart would otherwise zoom it instead.
                st.plotly_chart(fig, key="space_chart_%d" % i, config={"displaylogo": False, "scrollZoom": False})
                if missing:  # below the chart, so side-by-side charts stay aligned
                    st.caption("Not in this vocabulary: %s." % ", ".join(missing))
                tables.append(points.assign(embedding=name, x=coords[:, 0], y=coords[:, 1], z=coords[:, 2]))
        notes = []
        if method == "PCA":
            notes.append("Axis titles give the share of variance each principal component keeps.")
        if len(shown) > 1:
            notes.append("Each embedding is projected on its own, so compare clusters and shapes, not positions.")
        if notes:
            st.caption(" ".join(notes))
        if tables:
            with st.expander("Data table: plotted words and 3D coordinates"):
                table = pd.concat(tables, ignore_index=True)
                table = table[["embedding"] + [c for c in table.columns if c != "embedding"]]
                st.dataframe(table, hide_index=True, column_config={
                    c: st.column_config.NumberColumn(format="%.3f") for c in ("x", "y", "z", "cosine") if c in table})
                st.download_button("Download CSV", table.to_csv(index=False).encode("utf-8"), "embedding_space_3d.csv",
                                   "text/csv", key="space_download", icon=":material/download:")


def map_chart(ref, user_scores, level, title):
    """Paper curves (DP / non-DP per learning step) plus one dashed rule per loaded embedding."""
    colors = palette()
    data = ref[ref["level"] == level].copy()
    data["step"] = data["learning_step"].map(STEP_LABELS)
    domain = ["DP (paper)", "Non-DP (paper)"] + [u["name"] for u in user_scores]
    color_range = [colors[0], colors[1]] + [colors[u["slot"]] for u in user_scores]
    color = alt.Color("model:N", scale=alt.Scale(domain=domain, range=color_range),
                      legend=alt.Legend(title=None, orient="bottom", symbolStrokeWidth=2, labelLimit=320,
                                        columns=2))
    x = alt.X("step:O", sort=list(STEP_LABELS.values()), title="Learning step",
              axis=alt.Axis(labelAngle=0))
    # Character-level scores all sit near 0.9, so that axis zooms in instead of starting at 0.
    y = alt.Y("map:Q", title="MAP@100", axis=alt.Axis(format=".2f", tickCount=5),
              scale=alt.Scale(zero=level == "word"))
    tooltip = [alt.Tooltip("model:N", title="Model"), alt.Tooltip("step:O", title="Step"),
               alt.Tooltip("map:Q", title="MAP@100", format=".4f")]
    lines = alt.Chart(data).mark_line(strokeWidth=2).encode(x=x, y=y, color=color)
    points = alt.Chart(data).mark_point(filled=True, size=70, opacity=1).encode(x=x, y=y, color=color, tooltip=tooltip)
    layers = [lines, points]
    if user_scores:
        user_df = pd.DataFrame([{"model": u["name"], "map": u[level]} for u in user_scores])
        layers.append(alt.Chart(user_df).mark_rule(strokeWidth=2, strokeDash=[6, 4]).encode(
            y="map:Q", color=color,
            tooltip=[alt.Tooltip("model:N", title="Embedding"), alt.Tooltip("map:Q", title="MAP@100", format=".4f")]))
        # widen the hover target of the dashed rules
        layers.append(alt.Chart(user_df).mark_rule(strokeWidth=12, opacity=0).encode(
            y="map:Q", tooltip=[alt.Tooltip("model:N", title="Embedding"),
                                alt.Tooltip("map:Q", title="MAP@100", format=".4f")]))
    return alt.layer(*layers).properties(title=title, height=330)


with tab_semantic:
    st.markdown("For 11 frequent test words (*%s*), the 100 nearest neighbours in your embedding are compared "
                "with gold lists from a non-private word2vec model trained on text8, using MAP@100. Lines show the "
                "paper's DP and non-DP models at each learning step; dashed lines show your embeddings."
                % ", ".join(ev.TESTED_WORDS))
    ref = reference_curves()
    selected = []
    if ss.embeddings:
        # Only embeddings with a fixed chart colour (the first 6 loaded) can be plotted.
        colored = [n for n in embedding_options() if entry(n)["slot"] is not None]
        # Session state is the single source of truth for this widget (no `default=`): new loads
        # append to it in register(), and removed embeddings are dropped here.
        if "semantic_sel" not in ss:
            ss["semantic_sel"] = colored[:6]
        ss["semantic_sel"] = [n for n in ss["semantic_sel"] if n in colored][:6]
        selected = st.multiselect("Embeddings to score", colored, max_selections=6, key="semantic_sel")
        if len(colored) < len(ss.embeddings):
            st.caption("Up to 6 embeddings can be charted at once; remove one to chart the others.")
    user_scores, details = [], {}
    for name in selected:
        item = entry(name)
        if "semantic" not in item:
            item["semantic"] = ev.semantic_space_scores(item["kv"])
        per_word, neighbours = item["semantic"]
        found = per_word[per_word["in_vocabulary"]]
        user_scores.append({"name": item["name"], "slot": item["slot"],
                            "word": float(per_word["map_word"].mean()),
                            "character": float(per_word["map_character"].mean()),
                            "found": len(found)})
        details[item["name"]] = (per_word, neighbours)

    chart_cols = st.columns(2)
    chart_cols[0].altair_chart(map_chart(ref, user_scores, "word", "Word level"), width="stretch")
    chart_cols[1].altair_chart(map_chart(ref, user_scores, "character", "Character level"), width="stretch")

    if user_scores:
        summary = pd.DataFrame([{"Embedding": u["name"], "MAP@100 (word)": u["word"],
                                 "MAP@100 (character)": u["character"],
                                 "Test words in vocabulary": "%d / %d" % (u["found"], len(ev.TESTED_WORDS))}
                                for u in user_scores])
        st.dataframe(summary, hide_index=True, column_config={
            "MAP@100 (word)": st.column_config.NumberColumn(format="%.4f"),
            "MAP@100 (character)": st.column_config.NumberColumn(format="%.4f")})
        with st.expander("Per-word scores and neighbour lists"):
            pick = st.selectbox("Embedding", list(details.keys()), key="semantic_detail")
            per_word, neighbours = details[pick]
            st.dataframe(per_word, hide_index=True, column_config={
                "map_word": st.column_config.NumberColumn("MAP@100 (word)", format="%.4f"),
                "map_character": st.column_config.NumberColumn("MAP@100 (character)", format="%.4f"),
                "shared_with_gold": st.column_config.NumberColumn("Shared with gold top-100")})
            gold = ev.gold_lists()
            word = st.selectbox("Test word", [w for w in ev.TESTED_WORDS if w in neighbours] or ["-"], key="semantic_word")
            if word in neighbours:
                st.dataframe(pd.DataFrame({"rank": range(1, 21), "your embedding": neighbours[word][:20],
                                           "gold": gold[word][:20]}), hide_index=True)
    with st.expander("Data table: the paper's curves"):
        table = ref.pivot_table(index="learning_step", columns=["level", "model"], values="map")
        table = table.reindex(list(STEP_LABELS.keys()))
        table.columns = ["%s, %s level" % (model, level) for level, model in table.columns]
        table.index = [STEP_LABELS[s] for s in table.index]
        table.index.name = "Learning step"
        st.dataframe(table, column_config={c: st.column_config.NumberColumn(format="%.4f") for c in table.columns})


with tab_regression:
    st.markdown("Predicts the extraversion score of 250 Facebook users from their status updates "
                "(myPersonality). Features are character n-grams, word TF-IDF and the mean embedding of each user's "
                "words; unknown words fall back to the character embedding. The data is split 80/20 with the same "
                "seed as the paper and scored with RMSE (lower is better).")
    text_entries = [n for n in embedding_options() if entry(n)["format"] in ("word2vec_text", "text_no_header")
                    and not entry(n)["path"].lower().endswith(".gz")]
    none, shipped = NONE_OPTION, SHIPPED_CHAR_OPTION
    slot_cols = st.columns(3)
    word_slot = slot_cols[0].selectbox("Public word embedding", [none] + embedding_options(), key="reg_word",
                                       help="The paper used the GoogleNews vectors here.")
    char_slot = slot_cols[1].selectbox("Character embedding", [shipped] + text_entries, key="reg_char",
                                       help="Must be a plain-text file of character vectors.")
    private_slot = slot_cols[2].selectbox("Private (DP) embedding", [none] + embedding_options(), key="reg_private",
                                          help="The embedding trained on the private data, with or without DP.")
    fit_svr = st.checkbox("Also fit the linear SVR (slow: about 1-2 minutes)", key="reg_svr")
    if st.button("Run regression", type="primary", icon=":material/play_arrow:", key="reg_run"):
        word_kv = None if word_slot == none else entry(word_slot)["kv"]
        private_kv = None if private_slot == none else entry(private_slot)["kv"]
        char_path = es.DEFAULT_CHAR_EMBEDDING if char_slot == shipped else entry(char_slot)["path"]
        # Key on entry ids so a removed-and-reloaded embedding with the same name is not confused.
        cache_key = tuple(v if v in (none, shipped) else entry(v)["id"] for v in (word_slot, char_slot, private_slot))
        try:
            if cache_key not in ss.feature_cache:
                bar = st.progress(0.0, text="Building features...")
                ss.feature_cache[cache_key] = ev.build_personality_features(
                    word_kv, private_kv, char_path,
                    progress=lambda done, total: bar.progress(done / total, text="Embedding users %d / %d" % (done, total)))
                bar.empty()
            with st.spinner("Fitting %s..." % ("linear regression and SVR" if fit_svr else "linear regression")):
                rmse = ev.fit_regressors(*ss.feature_cache[cache_key], fit_svr=fit_svr)
            X_train = ss.feature_cache[cache_key][0]
            ss.regression_runs.append({
                "Run": len(ss.regression_runs) + 1,
                "Public word": word_slot,
                "Character": "shipped" if char_slot == shipped else char_slot,
                "Private (DP)": private_slot,
                "Features": X_train.shape[1],
                "LR RMSE": rmse["LR"],
                "SVR RMSE": rmse.get("SVR"),
            })
        except Exception as ex:
            st.error("Regression failed: %s: %s" % (type(ex).__name__, ex))
    if ss.regression_runs:
        st.dataframe(pd.DataFrame(ss.regression_runs), hide_index=True, column_config={
            "LR RMSE": st.column_config.NumberColumn(format="%.4f"),
            "SVR RMSE": st.column_config.NumberColumn(format="%.4f")})
        if st.button("Clear runs", key="reg_clear", icon=":material/delete_sweep:"):
            ss.regression_runs = []
            st.rerun()
    with st.expander("Reference: the paper's embeddings (cached features, linear regression)"):
        st.caption("Computed now from `data/saved/` with this environment's scikit-learn. These features used "
                   "GoogleNews + 100-d character vectors, so they are not directly comparable with runs above that "
                   "use other public embeddings; compare runs with and without a private embedding instead. "
                   "Linear regression here is underdetermined (200 users, ~2,000 features), so its RMSE moves "
                   "with library versions.")
        st.dataframe(reference_lr(), hide_index=True,
                     column_config={c: st.column_config.NumberColumn(format="%.4f")
                                    for c in ["Baseline (public only)", "+ DP embedding", "+ non-DP embedding"]})
