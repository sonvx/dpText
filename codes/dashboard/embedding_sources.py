"""Find, fetch and load word-embedding files for the dashboard.

Sources:
  * library  - files already inside the project (``data/embeddings/`` drop folder, the shipped public
               embeddings, and models trained by ``10.run_train_dp_embedding.sh`` under ``data/text8_data/``)
  * local    - any path on this machine (absolute, ``~/...``, relative to the repo, or a ``file://`` link)
  * upload   - a file sent from the browser (saved into ``data/embeddings/uploads/``)
  * url      - an http(s) direct link (downloaded into ``data/embeddings/downloads/``)

Formats: word2vec text (with ``<count> <dim>`` header), GloVe/fastText-style text without header,
word2vec binary (``.bin``), any of those gzipped (``.gz``), a ``.zip`` holding one of them, gensim
``KeyedVectors.save()`` files, and the folders written by this repo's training script
(``final_embeddings.pkl`` + ``words_dictionary.pkl``).

Formats that use Python pickle (gensim native files, the training-script folders) can execute code
when loaded, so they are only accepted from the library or a local path, never from uploads or URLs.
"""
import gzip
import os
import re
import shutil
import urllib.parse
import urllib.request
import zipfile
from dataclasses import dataclass

import numpy as np
from gensim.models import KeyedVectors

CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPO_DIR = os.path.dirname(CODES_DIR)
DATA_DIR = os.path.join(REPO_DIR, "data")

EMBEDDINGS_DIR = os.path.join(DATA_DIR, "embeddings")  # drop folder: copy embedding files here
UPLOADS_DIR = os.path.join(EMBEDDINGS_DIR, "uploads")
DOWNLOADS_DIR = os.path.join(EMBEDDINGS_DIR, "downloads")
EXTRACTED_DIR = os.path.join(EMBEDDINGS_DIR, "extracted")
LIBRARY_DIRS = [
    EMBEDDINGS_DIR,
    os.path.join(DATA_DIR, "pretrained_models"),
    os.path.join(DATA_DIR, "text8_data"),
]
DEFAULT_CHAR_EMBEDDING = os.path.join(DATA_DIR, "pretrained_models", "public_embeddings", "char-embeddings.txt")
TEXT8_SAMPLE_NAME = "text8_gensim_skipgram_200d.bin"  # built by dashboard/make_sample_embeddings.py

EMBEDDING_SUFFIXES = (".txt", ".vec", ".bin", ".w2v", ".emb", ".gz", ".zip", ".kv", ".model")
PICKLE_FORMATS = {"gensim_native", "dpugc_dir"}

FORMAT_LABELS = {
    "word2vec_text": "word2vec text (with header)",
    "text_no_header": "text, no header (GloVe/fastText style)",
    "word2vec_binary": "word2vec binary",
    "gensim_native": "gensim KeyedVectors (pickle)",
    "dpugc_dir": "dpUGC trained model folder (pickle)",
    "zip": "zip archive",
}

USER_AGENT = "dpUGC-embedding-dashboard/1.0"


class EmbeddingSourceError(Exception):
    """A user-facing problem with an embedding source (shown as-is in the dashboard)."""


@dataclass
class LibraryEntry:
    path: str
    label: str
    size_bytes: int


# --------------------------------------------------------------------------- helpers


def human_size(num_bytes):
    for unit in ["B", "KB", "MB", "GB"]:
        if num_bytes < 1024 or unit == "GB":
            return "%.0f %s" % (num_bytes, unit) if unit == "B" else "%.1f %s" % (num_bytes, unit)
        num_bytes /= 1024.0


def safe_filename(name, default="embedding.txt"):
    """Keep only a plain basename with safe characters (no directories, no traversal)."""
    name = os.path.basename(name.replace("\\", "/")).strip()
    name = re.sub(r"[^A-Za-z0-9._+-]", "_", name).lstrip(".")
    return name or default


def _unique_path(directory, filename):
    os.makedirs(directory, exist_ok=True)
    base, ext = os.path.splitext(filename)
    if ext == ".gz":  # keep double extensions like .txt.gz together
        base, ext2 = os.path.splitext(base)
        ext = ext2 + ext
    candidate, n = os.path.join(directory, filename), 1
    while os.path.exists(candidate):
        candidate = os.path.join(directory, "%s-%d%s" % (base, n, ext))
        n += 1
    return candidate


def is_dpugc_dir(path):
    return (os.path.isdir(path)
            and os.path.isfile(os.path.join(path, "final_embeddings.pkl"))
            and os.path.isfile(os.path.join(path, "words_dictionary.pkl")))


def file_uri(path):
    """A file:// link for a local path; it can be pasted back into the 'Local path or link' box."""
    return "file://" + urllib.parse.quote(os.path.abspath(path))


# --------------------------------------------------------------------------- sources


def list_library():
    """Embedding files and trained-model folders that already live inside the project."""
    entries, seen = [], set()
    for root_dir in LIBRARY_DIRS:
        if not os.path.isdir(root_dir):
            continue
        for dirpath, dirnames, filenames in os.walk(root_dir):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".") and d != "logs")
            if is_dpugc_dir(dirpath):
                size = sum(os.path.getsize(os.path.join(dirpath, f)) for f in
                           ("final_embeddings.pkl", "words_dictionary.pkl"))
                entries.append(LibraryEntry(dirpath, os.path.relpath(dirpath, REPO_DIR) + "/  [trained model]", size))
                seen.add(os.path.join(dirpath, "word2vec.txt"))  # same vectors, prefer the folder
                continue
            for filename in sorted(filenames):
                path = os.path.join(dirpath, filename)
                if (filename.lower().endswith(EMBEDDING_SUFFIXES) and not filename.startswith(".")
                        and not filename.endswith(".part") and path not in seen
                        and filename.lower() != "readme.txt"):
                    entries.append(LibraryEntry(path, os.path.relpath(path, REPO_DIR), os.path.getsize(path)))
    return entries


def text8_sample_path():
    """Where make_sample_embeddings.py writes the word-level sample trained on data/text8.zip."""
    return os.path.join(EMBEDDINGS_DIR, TEXT8_SAMPLE_NAME)


def list_samples():
    """Library entries the dashboard loads when it opens.

    That is the shipped public embeddings, files copied into ``data/embeddings/`` and trained models,
    but not uploads, web downloads or files extracted from archives (those were added by hand), and
    not zip archives (they may hold several files).
    """
    added_by_hand = tuple(os.path.join(d, "") for d in (UPLOADS_DIR, DOWNLOADS_DIR, EXTRACTED_DIR))
    return [e for e in list_library()
            if not e.path.startswith(added_by_hand) and not e.path.lower().endswith(".zip")]


def resolve_local_path(text):
    """Turn a pasted path or file:// link into an existing absolute path."""
    text = (text or "").strip().strip('"').strip("'")
    if not text:
        raise EmbeddingSourceError("Paste a file path or a file:// link.")
    if text.lower().startswith("file://"):
        parsed = urllib.parse.urlparse(text)
        if parsed.netloc not in ("", "localhost"):
            raise EmbeddingSourceError("Only local file:// links are supported (got host %r)." % parsed.netloc)
        text = urllib.parse.unquote(parsed.path)
    path = os.path.expanduser(text)
    if not os.path.isabs(path):
        # Relative paths are resolved against the repository root (where data/ lives).
        path = os.path.join(REPO_DIR, path)
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise EmbeddingSourceError("Not found: %s" % path)
    if os.path.isdir(path) and not is_dpugc_dir(path):
        raise EmbeddingSourceError("%s is a folder but not a trained-model folder "
                                   "(expected final_embeddings.pkl and words_dictionary.pkl)." % path)
    return path


def save_upload(filename, data):
    """Store uploaded bytes in data/embeddings/uploads/ and return the new path."""
    path = _unique_path(UPLOADS_DIR, safe_filename(filename))
    with open(path, "wb") as f:
        f.write(data)
    return path


def normalize_url(url):
    """Accept common share links and turn them into direct-download links."""
    url = (url or "").strip()
    parsed = urllib.parse.urlparse(url)
    if parsed.scheme.lower() == "file":
        return url
    if parsed.scheme.lower() not in ("http", "https"):
        raise EmbeddingSourceError("Links must start with http://, https:// or file://")
    host = parsed.netloc.lower()
    if host.endswith("dropbox.com"):
        query = dict(urllib.parse.parse_qsl(parsed.query))
        query["dl"] = "1"
        url = urllib.parse.urlunparse(parsed._replace(query=urllib.parse.urlencode(query)))
    elif host == "github.com" and "/blob/" in parsed.path:
        url = urllib.parse.urlunparse(parsed._replace(netloc="raw.githubusercontent.com",
                                                      path=parsed.path.replace("/blob/", "/", 1)))
    elif host == "huggingface.co" and "/blob/" in parsed.path:
        url = urllib.parse.urlunparse(parsed._replace(path=parsed.path.replace("/blob/", "/resolve/", 1)))
    return url


def _filename_from_response(response, url):
    disposition = response.headers.get("Content-Disposition", "")
    match = re.search(r'filename\*?=(?:UTF-8\'\')?"?([^";]+)"?', disposition)
    if match:
        return safe_filename(urllib.parse.unquote(match.group(1)))
    return safe_filename(urllib.parse.unquote(os.path.basename(urllib.parse.urlparse(url).path)),
                         default="downloaded_embedding.txt")


def download(url, progress=None, chunk_size=1 << 20, timeout=60, max_bytes=None):
    """Download an http(s) link into data/embeddings/downloads/ and return the local path.

    progress(downloaded_bytes, total_bytes_or_None) is called after each chunk. A file:// link is
    simply resolved to its local path (nothing is copied).
    """
    url = normalize_url(url)
    if url.lower().startswith("file://"):
        return resolve_local_path(url)
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        response = urllib.request.urlopen(request, timeout=timeout)
    except Exception as ex:  # URLError, HTTPError, timeouts...
        raise EmbeddingSourceError("Could not open %s: %s" % (url, ex))
    with response:
        content_type = response.headers.get("Content-Type", "")
        if content_type.startswith("text/html"):
            raise EmbeddingSourceError("The link returned a web page, not a file. Use a direct download link "
                                       "(e.g. the 'raw' or 'download' link).")
        total = response.headers.get("Content-Length")
        total = int(total) if total and total.isdigit() else None
        if max_bytes and total and total > max_bytes:
            raise EmbeddingSourceError("File is %s, above the %s limit." % (human_size(total), human_size(max_bytes)))
        dest = _unique_path(DOWNLOADS_DIR, _filename_from_response(response, url))
        tmp = dest + ".part"
        done = 0
        try:
            with open(tmp, "wb") as out:
                while True:
                    chunk = response.read(chunk_size)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    if max_bytes and done > max_bytes:
                        raise EmbeddingSourceError("Download exceeded the %s limit." % human_size(max_bytes))
                    if progress:
                        progress(done, total)
            if total is not None and done != total:
                raise EmbeddingSourceError("Download incomplete (%s of %s)." % (human_size(done), human_size(total)))
            os.replace(tmp, dest)
        finally:
            if os.path.exists(tmp):
                os.remove(tmp)
    return dest


# --------------------------------------------------------------------------- formats


def list_zip_members(path):
    with zipfile.ZipFile(path) as zf:
        return [info.filename for info in zf.infolist()
                if not info.is_dir() and not os.path.basename(info.filename).startswith(".")
                and "__MACOSX" not in info.filename]


def extract_zip_member(path, member):
    """Extract one member of a zip into data/embeddings/extracted/ (flat, safe filename)."""
    with zipfile.ZipFile(path) as zf:
        if member not in zf.namelist():
            raise EmbeddingSourceError("%s is not in %s" % (member, os.path.basename(path)))
        zip_stem = safe_filename(os.path.splitext(os.path.basename(path))[0])
        dest = os.path.join(EXTRACTED_DIR, zip_stem + "__" + safe_filename(member))
        info = zf.getinfo(member)
        if os.path.exists(dest) and os.path.getsize(dest) == info.file_size:
            return dest  # already extracted
        os.makedirs(EXTRACTED_DIR, exist_ok=True)
        with zf.open(member) as src, open(dest + ".part", "wb") as out:
            shutil.copyfileobj(src, out, 1 << 20)
        os.replace(dest + ".part", dest)
    return dest


def _open_text(path):
    if path.lower().endswith(".gz"):
        return gzip.open(path, "rt", encoding="utf-8", errors="replace")
    return open(path, "r", encoding="utf-8", errors="replace")


def detect_format(path):
    lower = path.lower()
    if os.path.isdir(path):
        if is_dpugc_dir(path):
            return "dpugc_dir"
        raise EmbeddingSourceError("Unsupported folder: %s" % path)
    if lower.endswith(".zip"):
        return "zip"
    if lower.endswith((".kv", ".model")):
        return "gensim_native"
    if lower.endswith((".bin", ".bin.gz")):
        return "word2vec_binary"
    try:
        with _open_text(path) as f:
            first = f.readline()
    except (OSError, EOFError) as ex:
        raise EmbeddingSourceError("Cannot read %s: %s" % (os.path.basename(path), ex))
    if "\x00" in first:
        return "word2vec_binary"
    parts = first.split()
    if len(parts) == 2 and all(p.isdigit() for p in parts):
        return "word2vec_text"
    if len(parts) >= 3:
        return "text_no_header"
    raise EmbeddingSourceError("Could not recognise the format of %s (first line: %r)"
                               % (os.path.basename(path), first[:80]))


def _load_text_tolerant(path, limit=None):
    """Parse whitespace-separated text vectors, with or without a header, skipping malformed rows.

    Rows whose word is itself whitespace (it disappears in split()) or that otherwise have the wrong
    number of values are skipped; duplicate words keep their first vector.
    """
    words, rows, dim = [], [], None
    seen = set()
    with _open_text(path) as f:
        for line_no, line in enumerate(f):
            # Split on plain spaces so words containing other unicode whitespace survive.
            parts = [p for p in line.rstrip("\r\n").split(" ") if p]
            if len(parts) < 3:
                parts = line.split()  # e.g. tab-separated files
            if line_no == 0 and len(parts) == 2 and all(p.isdigit() for p in parts):
                dim = int(parts[1])  # word2vec header
                continue
            if len(parts) < 3:
                continue
            if dim is None:
                dim = len(parts) - 1
            if len(parts) - 1 != dim:
                continue
            word = parts[0]
            if word in seen:
                continue
            try:
                vector = np.asarray(parts[1:], dtype=np.float32)
            except ValueError:
                continue
            seen.add(word)
            words.append(word)
            rows.append(vector)
            if limit and len(words) >= limit:
                break
    if not words:
        raise EmbeddingSourceError("No vectors found in %s" % os.path.basename(path))
    kv = KeyedVectors(vector_size=dim)
    kv.add_vectors(words, np.vstack(rows))
    return kv


def _load_dpugc_dir(path, limit=None):
    from word2vec import embedding_utils
    final_embeddings, _, words = embedding_utils.reload_embeddings(path)
    final_embeddings = np.asarray(final_embeddings, dtype=np.float32)
    if limit:
        words, final_embeddings = words[:limit], final_embeddings[:limit]
    kv = KeyedVectors(vector_size=final_embeddings.shape[1])
    kv.add_vectors(list(words), final_embeddings)
    return kv


def load_keyed_vectors(path, fmt=None, limit=None, allow_pickle=False):
    """Load embeddings as gensim KeyedVectors. ``limit`` keeps only the first N words."""
    fmt = fmt or detect_format(path)
    if fmt in PICKLE_FORMATS and not allow_pickle:
        raise EmbeddingSourceError("%s files use Python pickle, which can run code when loaded; they are only "
                                   "accepted from the project library or a local path." % FORMAT_LABELS[fmt])
    if fmt == "zip":
        raise EmbeddingSourceError("Pick a file inside the zip first.")
    if fmt == "word2vec_binary":
        return KeyedVectors.load_word2vec_format(path, binary=True, limit=limit, unicode_errors="ignore")
    if fmt == "word2vec_text":
        try:
            return KeyedVectors.load_word2vec_format(path, binary=False, limit=limit, unicode_errors="ignore")
        except (ValueError, IndexError, EOFError):
            return _load_text_tolerant(path, limit)  # e.g. rows with a whitespace "word"
    if fmt == "text_no_header":
        return _load_text_tolerant(path, limit)
    if fmt == "gensim_native":
        kv = KeyedVectors.load(path, mmap="r")
        if not isinstance(kv, KeyedVectors):
            kv = getattr(kv, "wv", None)
            if not isinstance(kv, KeyedVectors):
                raise EmbeddingSourceError("%s is not a gensim KeyedVectors/Word2Vec file." % path)
        return kv
    if fmt == "dpugc_dir":
        return _load_dpugc_dir(path, limit)
    raise EmbeddingSourceError("Unsupported format %r" % fmt)
