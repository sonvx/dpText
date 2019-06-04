import os
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import re
import emoji
from nltk.tokenize import TweetTokenizer
from word2vec import embedding_utils


class Text_Features(object):
    use_embedding = False
    word2vec_size = 300
    n_features = 4000
    n_lsi = 100
    random_state = 42
    n_iter = 100
    n_train = 3834
    n_list = [80, 100, 120]
    char2vec_size = 300
    debug = True
    saved_dir = None

    model_paths_list = None

    model_names_list = [
        "word2vec",
        "char2vec",
        "private_word2vec"
    ]

    model_dims_list = [
        300,
        100,
        300
    ]

    char_model_path = None
    char_model_dims = model_dims_list[1]

    def __init__(self):
        self.tokenizer = TweetTokenizer()

    def set_model_paths_list(self, paths_list, dim_list=None):
        self.model_paths_list =  paths_list
        self.char_model_path = paths_list[1]
        if dim_list:
            self.model_dims_list = dim_list

    @staticmethod
    def normalise_str(str_in):
        normalised_str = ""
        count = 0
        pre_char = None
        for i in range(len(str_in)):
            if i > 0:
                if str_in[i] == pre_char:
                    count += 1
                else:
                    count = 0
            if count <= 2:
                normalised_str += str_in[i]
            pre_char = str_in[i]
        return normalised_str

    @staticmethod
    def is_number(s):
        try:
            float(s)
            return True
        except ValueError:
            pass

        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass

        return False

    def enable_use_embedding(self, use_embedding):
        self.use_embedding = use_embedding

    def char_2_vec(self, word, char_model):
        chars = list(word)
        vecs = []
        for c in chars:
            if c in char_model:
                vecs.append(char_model[c])
        return list(vecs)

    def normalise_fb_status(self, tweet_str):

        tweet_str = (emoji.demojize(tweet_str))
        tweet_str = re.sub("\\s+", " ", re.sub("http.*?\\s", "url", tweet_str)
                           .replace(":", " ").replace("#", " #").replace("@", " @"))
        tweet = self.tokenizer.tokenize(tweet_str)
        normalised_tweet = ""

        for token_str in tweet:
            normalised_token_str = self.normalise_str(token_str.lower())
            if "haha" in normalised_token_str:
                token_str = "lol"
            if token_str.startswith("@"):
                normalised_tweet += "taggeduser "
            elif token_str.lower().startswith("http"):
                normalised_tweet += "url "
            elif self.is_number(token_str):
                normalised_tweet += "number "
            else: normalised_tweet += token_str + " "
        return normalised_tweet.strip().lower()

    # Convert text to vector using pretrained word2vec model
    def text_2_vec(self, tokenized_text, word_model, char_model):
        words = tokenized_text
        all_vecs = []
        char_vecs = []
        n_non_english_words = 0
        out_word_2_vec = []
        out_char_2_vec = []
        for word in words:
            if word in word_model and len(word) > 1:
                all_vecs.append(word_model[word]) # get word's vector.
            else:
                vecs = self.char_2_vec(word, char_model)
                # print("self.char_2_vec = ", type(vecs))
                if len(vecs) > 0:
                    char_vecs.extend(list(vecs))

        if len(all_vecs) > 0:
            out_word_2_vec = np.mean(all_vecs, axis=0)
        else:
            out_word_2_vec = np.zeros(self.word2vec_size)

        if len(char_vecs) > 0:
            out_char_2_vec = np.mean(char_vecs, axis=0)
        else:
            out_char_2_vec = np.zeros(self.char2vec_size)

        # print("DEBUG:", len(out_word_2_vec), len(out_char_2_vec), out_word_2_vec, out_char_2_vec)
        return out_word_2_vec, out_char_2_vec

    def tokenize_text(self, unicode_or_str, to_lowercase=False):
        import nltk
        # if type(text) == str:
        #     # print text
        #     text = unicode(text).encode('utf8')

        if isinstance(unicode_or_str, str):
            text = unicode_or_str
            decoded = False
        else:
            text = unicode_or_str.decode("utf8")
            decoded = True

        # print text
        # text = text.decode('utf-8')
        if to_lowercase:
            text = text.lower()
        text = re.sub("<.*?>|&.*?;", " ", text)
        # s = re.sub("\\W+", " ", s)
        tokenized_text = nltk.word_tokenize(text)
        return tokenized_text

    # Process for one product
    def extract_features_for_one_doc(self, status, model, char_model, custom_w2v_model):
        tokenized_desc = self.tokenize_text(status, True)

        output = [len(tokenized_desc), len(status)]

        if model is not None:
            # word_similarity = word_similarity_feature(tokenized_title, model)
            # output.append(word_similarity)
            word_2vec, char_2vec = self.text_2_vec(tokenized_desc, model, char_model)
            output.extend(word_2vec)
            output.extend(char_2vec)
        return output

    def load_char_model(self):
        char_model = dict()
        file_path = "pretrained_models/public_embeddings/char-embeddings.txt"
        file = open(file_path, "r")
        for line in file:
            elements = line.split()
            if len(elements) > 100: # because embedding dim is higher than 100.
                # char_model[elements[0]] = np.array(map(float, elements[1:])).tolist()
                char_model[elements[0]] = np.array([float(i) for i in elements[1:]]).tolist()
        return char_model

    def feature_generator(self, X_text, y_data, test_percent=0.2, load_from_file=False):
        """
        # 2. Regression based personality prediction
        :param X_text:
        :param train_valid_splitter:
        :param load_from_file:
        :return:
        """
        features = []

        # If not load from files
        if load_from_file is False:
            # Normalize text
            for idx in range(len(X_text)):
                X_text[idx] = self.normalise_fb_status(X_text[idx])

            # N-gram features
            ngram_vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(1, 5), min_df=1, lowercase=True, \
                                               max_features=1000, smooth_idf=True, norm='l2')

            counts = ngram_vectorizer.fit_transform(X_text)
            print("#N-gram char-based features = ", len(ngram_vectorizer.get_feature_names()))
            n_grams_features = counts.toarray()
            features = n_grams_features
            del n_grams_features

            # TF-IDF vectors
            tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 3), min_df=1, lowercase=True, \
                                               max_features=500, smooth_idf=True, norm='l2')
            tfidfs = tfidf_vectorizer.fit_transform(X_text)
            tfidfs_features = tfidfs.toarray()
            print("###INFO: word-based tfidfs_features = ", len(tfidfs_features))
            features = np.append(features, tfidfs_features, 1)
            print("#Tf-idf word-based Ngram features: ", len(tfidf_vectorizer.get_feature_names()))
            del tfidfs_features

            if self.use_embedding:
                print("INFO: using embeddings: ", self.model_names_list)

                char_model = embedding_utils.reload_char2vec_model(self.char_model_path, self.char_model_dims)
                embedding_models = embedding_utils.reload_embedding_models(self.model_paths_list,
                                                                           self.model_names_list,
                                                                           self.model_dims_list,
                                                                           char_model)

                embedding_features = []
                for text in X_text:
                    # TODO: matching this with input file
                    # embedding_feature = self.extract_features_for_one_doc(text, model, char_model)
                    # embedding_features.append(embedding_feature)
                    tokenized_text = self.tokenize_text(text, True)
                    output = [len(tokenized_text), len(text)]
                    doc_vector = embedding_models.get_vector_of_document(tokenized_text)
                    output.extend(doc_vector)

                    embedding_features.append(output)

                features = np.append(features, embedding_features, 1)

                print("Finished loading embedding data!")
                # del model
                # del char_model
                del embedding_models

            print("###INFO: final_#_feature =", len(features[0]))

            if test_percent != -1:
                print("features (%s), y_data (%s)" % (type(features), type(y_data)))
                X_train_data, X_test_data, y_train_data, y_test_data = \
                    train_test_split(features, y_data, test_size=test_percent, random_state=42)

                # X_train_data = np.asarray(features[0:train_valid_splitter])
                # X_test_data = np.asarray(features[train_valid_splitter:])

                # y_train_data = np.asarray(y_data[0:train_valid_splitter])
                # y_test_data = np.asarray(y_data[0:train_valid_splitter])
                print("Saving data ...")
                if not os.path.exists('saved'):
                    os.makedirs('saved')
                np.save(os.path.join(self.saved_dir, "X_train_data"), X_train_data)
                print("Saved X_train_data!")
                np.save(os.path.join(self.saved_dir, "X_test_data"), X_test_data)
                print("Saved X_test_data!")

                np.save(os.path.join(self.saved_dir, "y_train_data"), y_train_data)
                print("Saved y_test_data!")
                np.save(os.path.join(self.saved_dir, "y_test_data"), y_test_data)
                print("Saved y_test_data!")
                return X_train_data, X_test_data, y_train_data, y_test_data
            else:
                X_full_data = np.asarray(features)
                np.save(os.path.join(self.saved_dir, "X_full_data"), X_full_data)
                np.save(os.path.join(self.saved_dir, "y_full_data"), y_data)
                return X_full_data, y_data

        # If we load from saved data
        if load_from_file:
            if test_percent != -1:
                X_train_data = np.load(os.path.join(self.saved_dir, "X_train_data.npy"))
                print("Loaded train_data!")
                X_test_data = np.load(os.path.join(self.saved_dir, "X_test_data.npy"))
                print("Loaded test_data!")

                y_train_data = np.load(os.path.join(self.saved_dir, "y_train_data.npy"))
                print("Loaded y_test_data!")
                y_test_data = np.load(os.path.join(self.saved_dir, "y_test_data.npy"))
                print("Loaded y_test_data!")
                return X_train_data, X_test_data, y_train_data, y_test_data
            else:
                X_full_data = np.load(os.path.join(self.saved_dir, "X_full_data.npy"))
                print("Loaded X_full_data!")
                y_full_data = np.load(os.path.join(self.saved_dir, "y_full_data.npy"))
                print("Loaded y_full_data!")
                return X_full_data, y_full_data
