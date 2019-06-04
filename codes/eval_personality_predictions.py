import pandas as pd
from utils import dataset_utils
from feature_extractors import text_features
from sklearn.metrics import mean_squared_error, r2_score
import os

dir_root = "../data/saved/trained_models_dp_dpsgd/"
dp_w2v_models_dict = {
    "20": dir_root + "_20epoch/word2vec.txt",
    "200": dir_root + "_200epoch/word2vec.txt",
    "500": dir_root + "_500epoch/word2vec.txt",
    "1000": dir_root + "_1000epoch/word2vec.txt",
    "5000": dir_root + "_5000epoch/word2vec.txt",
    "10000": dir_root + "_10000epoch/word2vec.txt",
    "50000": dir_root + "_50000epoch/word2vec.txt",
    "90000": dir_root + "_90000epoch/word2vec.txt",
    "100000": dir_root + "_100000epoch/word2vec.txt"
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


def run_evaluation_on_embedding_type(models_dict, partial_path, emb_type):
    model_paths_list = [
        # Get it here: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
        "../data/pretrained_models/public_embeddings/GoogleNews-vectors-negative300.bin",
        "../data/pretrained_models/public_embeddings/char-embeddings.txt",
        ""  # Await position for the evaluated embedding. E.g., dp_embedding_20epoch.
    ]

    learning_steps_arr = []
    svm_rmse_results_arr = []
    lr_rmse_results_arr = []
    results = []
    for model_name in models_dict.keys():
        print("------ Working on embedding at %s learning step ------"%(model_name))
        model_path = models_dict[model_name]
        # private model file path
        model_paths_list[2] = model_path
        saved_dir = ("../data/saved/%s/w2v_%s_epochs" % (partial_path, model_name))
        X_train_data, y_train_data, X_test_data, y_test_data = load_data(saved_dir, model_paths_list,
                                                                         is_reload_from_file=True)
        result_msg = "\n######### model_name: %s##########\n" % model_name
        learning_steps_arr.append(model_name)
        svm_rmse_arr, lr_rmse_arr, result_str = run_traditional_ml_models(X_train_data,
                                                                          y_train_data,
                                                                          X_test_data,
                                                                          y_test_data)
        svm_rmse_results_arr.append(svm_rmse_arr[0])
        lr_rmse_results_arr.append(lr_rmse_arr[0])
        result_msg += result_str
        results.append(result_msg)
    print("####" * 10)
    # print(results)
    result_df = pd.DataFrame(columns=['Learning_Steps', emb_type + '-SVR', emb_type + '-LR'])
    result_df['Learning_Steps'] = learning_steps_arr
    result_df[emb_type + '-SVR'] = svm_rmse_results_arr
    result_df[emb_type + '-LR'] = lr_rmse_results_arr
    # print(result_df)
    # print("####" * 10)
    return result_df


def run_main():
    # 01.
    print("01. Run evaluation on LR task using Embedding(private with privacy guarantee) + Embedding(public):")
    dp_results_df = run_evaluation_on_embedding_type(dp_w2v_models_dict, "saved_dp", "DP")

    # 02.
    print("02. Run evaluation on LR task using Embedding(private without privacy guarantee) + Embedding(public):")
    no_dp_results_df = run_evaluation_on_embedding_type(nodp_w2v_models_dict, "saved_no_dp", 'NoneDP')

    # 03.
    final_results_df = pd.DataFrame(columns=['Learning_Steps',
                                             'Baseline-SVR', 'DP-SVR', 'NoneDP-SVR',
                                             'Baseline-LR',  'DP-LR', 'NoneDP-LR'])

    print("03. Run evaluation on LR task using Embedding(public) only:")
    saved_dir = "../data/saved/w2v_char2vec_no_private2v"
    model_paths_list = [
        # Get it here: https://github.com/mmihaltz/word2vec-GoogleNews-vectors
        "../data/pretrained_models/public_embeddings/GoogleNews-vectors-negative300.bin",
        "../data/pretrained_models/public_embeddings/char-embeddings.txt",
    ]
    X_train_data, y_train_data, X_test_data, y_test_data = load_data(saved_dir, model_paths_list,
                                                                     is_reload_from_file=True)
    svm_rmse_arr, lr_rmse_arr, result_msg = run_traditional_ml_models(X_train_data,
                                                                      y_train_data, X_test_data, y_test_data)

    # Constructing final DF
    final_results_df['Learning_Steps'] = dp_results_df['Learning_Steps']
    final_results_df['Baseline-SVR'] = svm_rmse_arr*len(final_results_df['Learning_Steps'])
    final_results_df['Baseline-LR'] = lr_rmse_arr * len(final_results_df['Learning_Steps'])
    final_results_df['DP-SVR'] = dp_results_df['DP-SVR']
    final_results_df['DP-LR'] = dp_results_df['DP-LR']
    final_results_df['NoneDP-LR'] = no_dp_results_df['NoneDP-LR']
    final_results_df['NoneDP-SVR'] = no_dp_results_df['NoneDP-SVR']

    # print(result_msg)
    print("####" * 10)
    print(final_results_df)
    print("####" * 10)


if __name__ == "__main__":
    run_main()

