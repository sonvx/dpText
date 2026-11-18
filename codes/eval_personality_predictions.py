import argparse
import os

import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import mean_squared_error, r2_score

from utils import dataset_utils
from feature_extractors import text_features

# Public embeddings shared by every run.
# GoogleNews vectors: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
# (only needed when features are regenerated, i.e. not loaded from ../data/saved/).
PUBLIC_MODEL_PATHS = [
    "../data/pretrained_models/public_embeddings/GoogleNews-vectors-negative300.bin",
    "../data/pretrained_models/public_embeddings/char-embeddings.txt",
]

dir_root = "../data/saved/trained_models_dp_dpsgd/"
dp_w2v_models_dict = {
    "20": dir_root + "_20epoch/",
    "200": dir_root + "_200epoch/",
    "500": dir_root + "_500epoch/",
    "1000": dir_root + "_1000epoch/",
    "5000": dir_root + "_5000epoch/",
    "10000": dir_root + "_10000epoch/",
    "50000": dir_root + "_50000epoch/",
    "90000": dir_root + "_90000epoch/",
    "100000": dir_root + "_100000epoch/"
}

dir_root = "../data/saved/trained_models_nodp_dpsgd/"
nodp_w2v_models_dict = {
    "20": dir_root + "_20epoch/",
    "200": dir_root + "_200epoch/",
    "500": dir_root + "_500epoch/",
    "1000": dir_root + "_1000epoch/",
    "5000": dir_root + "_5000epoch/",
    "10000": dir_root + "_10000epoch/",
    "50000": dir_root + "_50000epoch/",
    "90000": dir_root + "_90000epoch/",
    "100000": dir_root + "_100000epoch/"
}


def load_data(saved_dir, model_paths_list, is_reload_from_file=True):
    """

    :param saved_dir:
    :param model_paths_list:
    :param is_reload_from_file:
    :return:
    """
    text_features_obj = text_features.Text_Features()
    text_features_obj.set_model_paths_list(model_paths_list)
    text_features_obj.enable_use_embedding(True)
    text_features_obj.saved_dir = saved_dir
    if not os.path.exists(text_features_obj.saved_dir):
        os.makedirs(text_features_obj.saved_dir)

    # 1. Load data
    raw_data_df = pd.read_csv("../data/personality/Full_Data.csv", encoding="utf-8", sep=';')
    list_document, labels_list, authorIds, sEXT_values = dataset_utils.process_personality_data(raw_data_df)

    print("y_values = ", type(sEXT_values[0]), sEXT_values[:5])
    test_percent = 0.2

    X_train_data, X_test_data, y_train_data, y_test_data = \
        text_features_obj.feature_generator(list_document, sEXT_values, test_percent,
                                            load_from_file=is_reload_from_file)
    print("Done with load data")

    return X_train_data, y_train_data, X_test_data, y_test_data


def run_traditional_ml_models(X_train_data, y_train_data, X_test_data, y_test_data):
    """

    :return:
    """
    from sklearn import svm, linear_model

    svm_classifier = svm.SVR(kernel='linear', C=1, epsilon=0.2)
    lr_clf = linear_model.LinearRegression()
    clf_names = ['SVR', 'LR']

    list_clfs = [svm_classifier, lr_clf]

    returned_results = ""
    svm_rmse_arr = []
    lr_rmse_arr = []
    for clf_name, clf in zip(clf_names, list_clfs):
        clf.fit(X_train_data, y_train_data)
        y_pred = clf.predict(X_test_data)
        r2_score_Variance = r2_score(y_test_data, y_pred)
        rmse = mean_squared_error(y_test_data, y_pred) ** 0.5
        if clf_name == 'SVR':
            svm_rmse_arr.append(rmse)
        elif clf_name == 'LR':
            lr_rmse_arr.append(rmse)

        # The coefficient R^2 is defined as (1 - u/v), where u is the residual sum of squares
        # ((y_true - y_pred) ** 2).sum()
        # and v is the total sum of squares ((y_true - y_true.mean()) ** 2).sum().
        # msg = "%s \n R^2 Score (1 - u/v): %0.4f, \nRMSE = %0.4f"%(clf_name, r2_score_Variance, rmse)
        msg = "Classifier: %s,  RMSE = %0.4f" % (clf_name, rmse)
        returned_results += msg + "\n"

    return svm_rmse_arr, lr_rmse_arr, returned_results


def evaluate_one_embedding(model_paths_list, saved_dir):
    """Load (cached) features for one embedding configuration and return (SVR RMSE, LR RMSE)."""
    X_train_data, y_train_data, X_test_data, y_test_data = load_data(saved_dir, model_paths_list,
                                                                     is_reload_from_file=True)
    svm_rmse_arr, lr_rmse_arr, _ = run_traditional_ml_models(X_train_data, y_train_data,
                                                             X_test_data, y_test_data)
    return svm_rmse_arr[0], lr_rmse_arr[0]


def run_main(n_jobs=-1):
    # Every (embedding, learning step) pair is an independent model fit. The linear-kernel SVR takes
    # ~1-2 minutes per fit (the two raw length features are unscaled, so libsvm converges slowly), so
    # the 19 fits are run in parallel. Results are identical to running them sequentially.
    jobs = []  # (emb_type, learning_step, model_paths_list, saved_dir)
    for emb_type, models_dict, partial_path in [("DP", dp_w2v_models_dict, "saved_dp"),
                                                ("NoneDP", nodp_w2v_models_dict, "saved_no_dp")]:
        for model_name, model_path in models_dict.items():
            saved_dir = "../data/saved/%s/w2v_%s_epochs" % (partial_path, model_name)
            jobs.append((emb_type, model_name, PUBLIC_MODEL_PATHS + [model_path], saved_dir))
    # Embedding(public) only:
    jobs.append(("Baseline", None, list(PUBLIC_MODEL_PATHS), "../data/saved/w2v_char2vec_no_private2v"))

    print("01. Run evaluation on LR task using Embedding(private with privacy guarantee) + Embedding(public)")
    print("02. Run evaluation on LR task using Embedding(private without privacy guarantee) + Embedding(public)")
    print("03. Run evaluation on LR task using Embedding(public) only")
    print("Fitting %d SVR + LR models (n_jobs=%s) ..." % (len(jobs), n_jobs))
    scores = Parallel(n_jobs=n_jobs)(
        delayed(evaluate_one_embedding)(paths, saved_dir) for _, _, paths, saved_dir in jobs)

    learning_steps = list(dp_w2v_models_dict.keys())
    results = {}
    for (emb_type, model_name, _, _), (svr_rmse, lr_rmse) in zip(jobs, scores):
        results[(emb_type, model_name)] = (svr_rmse, lr_rmse)
    baseline_svr, baseline_lr = results[("Baseline", None)]

    final_results_df = pd.DataFrame({
        'Learning_Steps': learning_steps,
        'Baseline-SVR': [baseline_svr] * len(learning_steps),
        'DP-SVR': [results[("DP", step)][0] for step in learning_steps],
        'NoneDP-SVR': [results[("NoneDP", step)][0] for step in learning_steps],
        'Baseline-LR': [baseline_lr] * len(learning_steps),
        'DP-LR': [results[("DP", step)][1] for step in learning_steps],
        'NoneDP-LR': [results[("NoneDP", step)][1] for step in learning_steps],
    })

    print("####" * 10)
    print(final_results_df.to_string())
    print("####" * 10)
    return final_results_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Reproduce Table 3 (personality regression) of the dpUGC paper.")
    parser.add_argument("--n_jobs", type=int, default=-1,
                        help="Number of model fits to run in parallel (-1 = all CPU cores, 1 = sequential).")
    run_main(n_jobs=parser.parse_args().n_jobs)
