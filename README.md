# dpUGC: Learn Differentially Private Representation for User Generated Contents 
* Paper: https://arxiv.org/abs/1903.10453
## How to cite:
```
@InProceedings{Vu:CiCLing:2019b,
	author    = {Xuan-Son Vu, Son N. Tran, Lili Jiang},
	title     = {dpUGC: Learn Differentially Private Representation for User Generated Contents},
	booktitle   = {Proceedings of the 20th International Conference on Computational Linguistics and Intelligent Text Processing, April, 2019},
	year      = {2019},
	location = 	{La Rochelle, France}
}
```

## How to prepare the environment to run the code:
Requires **Python 3.10 – 3.13** (Python 3.14 is not supported yet because TensorFlow and gensim
do not publish wheels for it). Create a python environment using
[venv](https://docs.python.org/3/library/venv.html) or [anaconda](https://www.anaconda.com/distribution/),
then install the dependencies in that environment:
```bash
python3.13 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```
TensorFlow is only needed to train new embeddings; the two evaluation experiments below run without it.
The code was originally written for TensorFlow 1.12 and now runs on TensorFlow 2.x through `tf.compat.v1`.

To check the installation (this also checks that Experiment 1 reproduces Figure 2 exactly;
the dashboard tests are skipped unless `requirements-dashboard.txt` is installed):
```bash
cd codes/
python3 -m unittest discover -s tests -t .
```

## Embedding dashboard (local web app)
A small web app to load word embeddings and evaluate them with the paper's two experiments.
```bash
pip install -r requirements-dashboard.txt   # requirements.txt + Streamlit, Altair, Plotly
./codes/run_dashboard.sh                    # then open http://localhost:8501
```
(`run_dashboard.sh` uses `./.venv` automatically if it exists and no environment is active.)

**Project samples load automatically** when the dashboard opens: everything in the project library except
uploads, downloads and zip archives (at most 6, the first 100,000 words of each). The repository ships the
character embeddings; build a word-level sample (gensim skip-gram, 200 dimensions, 50,000 words, trained on
`data/text8.zip` without DP, about a minute) with the *Build the text8 sample* button in the Project library tab,
or from a terminal:
```bash
cd codes && python -m dashboard.make_sample_embeddings
```
Set `DPUGC_AUTOLOAD_SAMPLES=0` to start with nothing loaded.

**Add embeddings** in three ways:
* **Paste a path or link**: a local file path (absolute, `~/...` or relative to the repository), a `file://` link,
  or an `http(s)://` direct download link. Local files are loaded in place without copying; web files are
  downloaded to `data/embeddings/downloads/`. Dropbox, GitHub and Hugging Face "view" links are turned into
  direct-download links automatically.
* **Upload a file** (up to 1 GB, saved to `data/embeddings/uploads/`).
* **Project library**: anything you copy into `data/embeddings/`, the shipped public embeddings, and models
  trained with `10.run_train_dp_embedding.sh`.

Supported formats: word2vec text (with header), GloVe/fastText-style text, word2vec binary (`.bin`), any of
these gzipped (`.gz`) or inside a `.zip`, gensim `KeyedVectors.save()` files, and the model folders written by
the training script. Pickle-based formats (gensim native, training-script folders) can execute code when
loaded, so they are only accepted from a local path or the project library, not from uploads or web links.
Use *Max words to load* to keep only the most frequent words of a very large file (e.g. GoogleNews).

**Explore and evaluate**:
* *Explore neighbours*: nearest neighbours, word similarity, and neighbour overlap between two embeddings
  (e.g. a DP model and its non-DP twin).
* *3D embedding space*: an interactive 3D scatter (PCA or t-SNE) of chosen words and their neighbours, one
  colour per word, or of the most frequent words with highlighted words labelled. Compare two embeddings side by
  side; with *Same words in both*, the first embedding's neighbourhoods are drawn in the second one too, so you
  can see whether its clusters survive (e.g. under DP). The coordinates can be downloaded as CSV.
* *Semantic space (Figure 2)*: MAP@100 of each embedding's neighbours against the gold lists, at word and
  character level, plotted against the paper's DP / non-DP curves.
* *Personality regression (Table 3)*: choose the public word, character and private (DP) embeddings, then
  fit linear regression (and optionally the slow linear SVR) on the myPersonality data; runs are kept in a
  table for comparison.

The server only listens on `localhost` because it can read any local file you point it at.

## How to train privacy-guarantee embedding models on new dataset.
* This is for training private embedding on new data:
> cd codes/ <br>
> ./10.run_train_dp_embedding.sh

* The two runs (with and without DP) train for 100,001 steps each by default and save a snapshot
(embeddings in word2vec format, checkpoint, t-SNE plot and the privacy budget spent) at each checking point
under `data/text8_data/`. Pass extra flags to shorten a run, e.g. `./10.run_train_dp_embedding.sh --NUM_STEPS=1001`.

## How to reproduce experiments in the paper:

### Experiment 1: Semantic Changes
* How to run:<br>
> cd codes/ <br>
>
> ./01.run_changes_in_semantic_spaces.sh
* Expected outputs: see the evaluation results from the console (see codes/images/results_fig2.png). 
The results should be similar to **Figure 2** in the paper.

* Note:
It takes time to train the embedding model so we already extracted the top similarity 
words to run the evaluation. If it's needed, users are able to train again from the text8 corpus
and select the top similar words and run this evaluation again.

### Experiment 2: Linear Regression Task.
* How to run:
> cd codes/ <br>
>
> ./02.run_evaluation_regression_task.sh

* Expected outputs: printed in the console (see codes/images/results_table3.png). Evaluation results similar to **Table 3** in the paper. 
The linear-kernel SVR converges slowly on these features (about 1.5 minutes per fit), so the 19 model fits run
in parallel on all CPU cores by default; use `./02.run_evaluation_regression_task.sh --n_jobs 1` to run them one by one.

Note that the privacy-budget column was stated after the training embedding with differential privacy, 
i.e., one according privacy-budget (\delta, \epsilon) is calculated at each checking point.
