from gensim.models import KeyedVectors
from utils import chart_utils


constants_dict = {
    "top_similarity_topN": 100, #8,
    "top_similarity_num_tested_words": 11
}


def evaluate_semantic_spaces_topN(models_dict, outfile_name):
    """
    Given list of queried words, for each word, we get its top 100 similar words and write to a file.
    :param models_dict: embedding model in word2vec's object
    :param outfile_name: list of 100 similar words to each word in the given tested_words.
    :return: to the output file.
    """
    import pandas as pd
    tested_words = ['been', 'time', 'over', 'not', 'three', 'between', 'from', 'eight', 'they', 'more', 'other']

    # 1. Get topN gold outputs of 10 words from nonEmbedding model
    topN = 100
    columns_names = ['model_name', 'word']
    for i in range(topN):
        columns_names.append("top_%s" % (i + 1))

    print("len(columns_names) = ", len(columns_names))

    final_df = pd.DataFrame(columns=columns_names)
    print("final_df = ", final_df.head())
    iloc = 0
    for model_name in models_dict.keys():
        path_to_model1 = models_dict[model_name]
        print("Loading ...", path_to_model1)
        word2v_model = KeyedVectors.load_word2vec_format(path_to_model1, binary=False)
        # working on each tested word
        for tested_word in tested_words:
            # get top similar words to the tested word
            top_sim = word2v_model.most_similar(tested_word, topn=topN)
            get_top_sim_words_arr = [model_name, tested_word]
            for word_similar in top_sim:
                get_top_sim_words_arr.append(word_similar[0])  # + ' - ' + str(wordsimilar[1]))
            final_df.loc[iloc] = get_top_sim_words_arr
            iloc += 1

    final_df.to_csv("../data/evaluate_embeddings/topsimilarity100" + outfile_name + ".csv")
    return None


def calculate_map(task_name, with_gold_results=False, is_word_level=False):
    import pandas as pd
    from utils import eval_utils
    import numpy as np
    # no_dp_df = "../Data/EvaluatedResults/nodp_w2v_models.csv"
    # dp_df = "../Data/EvaluatedResults/dp_w2v_models.csv"

    if task_name == "word_similarity": # top similarity
        # dp_df = "../Data/EvaluatedResults/dp_w2v_models.csv"
        # no_dp_df = "../Data/EvaluatedResults/nodp_w2v_models.csv"
        # gold_values_file = "../Data/EvaluatedResults/Text8_Gold_TopWords.csv"
        dp_df = "../data/evaluate_embeddings/topsimilarity100/dp_w2v_models.csv"
        no_dp_df = "../data/evaluate_embeddings/topsimilarity100/nodp_w2v_models.csv"
        gold_values_file = "../data/evaluate_embeddings/topsimilarity100/topsimilarity_100_gold.csv"

        topN = constants_dict["top_similarity_topN"]
        num_tested_words = constants_dict["top_similarity_num_tested_words"]
    else:
        raise Exception("You must specify task name: word_similarity or word_analogies")

    if with_gold_results:
        gold_values_df = pd.read_csv(gold_values_file)

    def map_to_gold_values(predicted_df, gold_df, num_tested_words):
        list_map_results = []
        new_model_counter = 1
        model_lists = []
        gold_value_idx = 0
        # testing NoDP models:
        for idx in predicted_df.index:
            if new_model_counter <= num_tested_words:
                gold_words = gold_df.loc[gold_value_idx].tolist()
                predicted_words = predicted_df.loc[idx].tolist()
                value = eval_utils.mapk(gold_words, predicted_words, k=topN, word_level=is_word_level)
                model_lists.append(value)

            if new_model_counter % num_tested_words == 0:
                # reset index for preditect DF
                new_model_counter = 1
                # reset index for gold standard
                gold_value_idx = 0
                list_map_results.append(np.mean(model_lists))
                model_lists = []
            else:
                new_model_counter += 1
                gold_value_idx += 1
        return list_map_results

    evaluated_columns = []
    for i in range(topN):
        evaluated_columns.append("top_%s" % (i + 1))

    nodp_df = pd.read_csv(no_dp_df)
    nodp_values = nodp_df[evaluated_columns]

    dp_df = pd.read_csv(dp_df)
    dp_values = dp_df[evaluated_columns]

    # compare two method with gold values
    if with_gold_results:
        gold_evaluated_values_df = gold_values_df[evaluated_columns]
        list_01 = map_to_gold_values(nodp_values, gold_evaluated_values_df, num_tested_words)
        list_02 = map_to_gold_values(dp_values, gold_evaluated_values_df, num_tested_words)
        print("Results (NoDP vs. Real-Gold = ", list_01)
        print("Results (DP vs. Real-Gold = ", list_02)

        # if is_word_level:
        #     chart_utils.draw_line_charts_fig2left_wordlevel(list_01, list_02)
        # else:
        #     chart_utils.draw_line_charts_fig2right_charlevel(list_01, list_02)

    else: # compare two method with each others.
        list_results = map_to_gold_values(dp_values, nodp_values, num_tested_words)
        print("Results (DP vs. NoDP-As-Gold = ", list_results)


if __name__ == "__main__":
    # Run evaluation
    print("========= Fig.2 (left) =========")
    print("Evaluating MAP at Word-Level:")
    calculate_map(task_name="word_similarity", with_gold_results=True, is_word_level=True)
    print("========= Fig.2 (right) =========")
    print("Evaluating MAP at Character-Level:")
    calculate_map(task_name="word_similarity", with_gold_results=True, is_word_level=False)
