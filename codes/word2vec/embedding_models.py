from gensim.models import KeyedVectors as Word2Vec
import numpy as np
from word2vec import embedding_utils
from utils import file_utils
import collections
import os, re
import logging


DEBUG = False


class model_constants(object):
    word2vec = "word2vec"
    char2vec = "char2vec"
    private_word2vec = "private_word2vec"
    elmo = "elmo"


class embedding_model(object):
    def __init__(self, name, vector_dim):
        self.name = name
        self.model = None
        self.char_model = None
        self.vocabs_list = None
        self.vocab_index = None  # word -> row index, built from vocabs_list for O(1) lookups
        self.vector_dim = vector_dim
        # TODO: update this changeable param later
        # unk, random, mean, replace_by_character_embedding
        self.unknown_word = "replace_by_character_embedding"
        # self.MAX_DIM = 400 # No longer use MAX_DIM, now it depends on input dims

    def load_model(self, model_path):
        if self.name == model_constants.word2vec or self.name == model_constants.elmo:
            if model_path.endswith(".bin"):
                self.model = Word2Vec.load_word2vec_format(model_path, binary=True)
            else:
                self.model = Word2Vec.load_word2vec_format(model_path, binary=False)
        elif self.name == model_constants.char2vec:
            self.model = dict()
            print("Loading model_path = ", model_path)
            with open(model_path, "r", encoding="utf-8") as file:
                for line in file:
                    elements = line.split()
                    if len(elements) > 100:  # because embedding dim is higher than 100.
                        self.model[elements[0]] = [float(i) for i in elements[1:]]
            # Rows whose character is whitespace lose their key in split() and come out one value
            # short; drop rows that don't have the file's dominant dimension.
            dims = collections.Counter(len(vec) for vec in self.model.values())
            if dims:
                file_dim = dims.most_common(1)[0][0]
                self.model = {c: vec for c, vec in self.model.items() if len(vec) == file_dim}
                if file_dim != self.vector_dim:
                    # E.g. the shipped char-embeddings.txt is 300-d while the default config says 100.
                    logging.warning("char2vec: using the file's dimension %s instead of the configured %s (%s)",
                                    file_dim, self.vector_dim, model_path)
                    self.vector_dim = file_dim
            return self.model
        elif self.name == model_constants.private_word2vec:
            self.model, _, vocabs_list = embedding_utils.reload_embeddings(model_path)
            self._set_vocabs_list(vocabs_list)
        else:
            raise Exception("Unknown embedding models!")

    def is_punct(self, word):
        arr_list = [
            '!',
            '"',
            '%',
            '&',
            "'",
            "''",
            '(',
            '(.',
            ')',
            '*',
            '+',
            ',',
            '-',
            '---',
            '.',
            '..',
            '...',
            '....',
            '/',
            ]
        if word in arr_list:
            return True
        else:
            return False

    def is_number(self, word):
        regex = r"^[0-9]+"
        matches = re.finditer(regex, word, re.MULTILINE)
        matchNum = 0
        for matchNum, match in enumerate(matches):
            matchNum = matchNum + 1
        if matchNum > 0:
            return True
        else:
            return False

    def _set_vocabs_list(self, vocabs_list):
        self.vocabs_list = vocabs_list
        self.vocab_index = None
        if vocabs_list is not None:
            self.vocab_index = {}
            for idx, word in enumerate(vocabs_list):
                self.vocab_index.setdefault(word, idx)  # keep first occurrence, like list.index()

    def set_char_model(self, char_model):
        self.char_model = char_model

    def load_vocabs_list(self, vocab_file_path):
        """
        Load vocabs list for private w2v model. Has to be pickle file.
        :param vocab_file_path:
        :return:
        """
        if vocab_file_path:
            self._set_vocabs_list(file_utils.load_obj(vocab_file_path))

    def get_char_vector(self, char_model, word):
        """
        char_model here is an instance of embedding_model
        :param char_model: an instance of embedding_model
        :param word:
        :return:
        """
        if char_model is None:
            raise Exception("Char_model is None! Cannot use character-embedding.")

        out_char_2_vec = []
        char_vecs = []
        chars = list(word)
        vecs = []
        for c in chars:
            if c in char_model.model:
                emb_vector = char_model.model[c]
                vecs.append(emb_vector)
                if DEBUG:
                    input(">>>>>>")
                    print("Char_emb_vector=", emb_vector)

        # char_vecs.extend(list(vecs))

        if len(vecs) > 0:
            out_char_2_vec = np.mean(vecs, axis=0)

        if DEBUG:
            print(">>> Output of char2vec: %s"%(out_char_2_vec))
            input(">>>> outc2v ...")

        return out_char_2_vec

    def is_unknown_word(self, word):
        """Check whether or not a word is unknown"""
        is_unknown_word = False
        if self.vocab_index is not None:
            if word not in self.vocab_index:
                is_unknown_word = True
        else:
            if word not in self.model:
                is_unknown_word = True
        return is_unknown_word

    def get_word_vector(self, word):
        """
        Handle unknown word: In case of our private word2vec, we have a vocabs_list to check. With regular models,
        we can check inside the model. Note that by default, we use char-model to handle unknown words.
        :param word:
        :param char_model:
        :return:
        """

        rtn_vector = []

        # try first time with normal case
        is_unknown_word = self.is_unknown_word(word)

        # try 2nd times with owercase.
        if is_unknown_word:
            word = word.lower()
            is_unknown_word = self.is_unknown_word(word)

        # unknown word
        if is_unknown_word:
            rtn_vector = self.get_vector_of_unknown(word)
        else:
            # normal case
            if self.name == model_constants.word2vec:
                rtn_vector = self.model[word]

                # For now we have self.vector_dim, max_dim, and len(rtn_vector)
                # Update: move to use self.vector_dim only
                if len(rtn_vector) > self.vector_dim:
                    logging.debug("auto trim to %s/%s dimensions", self.vector_dim, len(rtn_vector))
                    rtn_vector = self.model[word][:self.vector_dim]

            elif self.name == model_constants.elmo:
                rtn_vector = self.model[word]

                if self.vector_dim == len(rtn_vector)/2:
                    vector1 = rtn_vector[:self.vector_dim]
                    vector2 = rtn_vector[self.vector_dim:]
                    print("Notice: auto average to  b[i] = (a[i] + a[i + %s])/2 /%s dimensions" % (self.vector_dim,
                                                                                                   len(rtn_vector)))
                    rtn_vector = np.mean([vector1, vector2], 0)
                elif len(rtn_vector) > self.vector_dim:
                    logging.debug("auto trim to %s/%s dimensions", self.vector_dim, len(rtn_vector))
                    rtn_vector = self.model[word][:self.vector_dim]

            elif self.name == model_constants.char2vec:
                rtn_vector = self.get_char_vector(self, word)

            elif self.name == model_constants.private_word2vec:
                # Handle unknown word - Not need for now since we handle unknown words first
                if word not in self.vocab_index:
                    word = "UNK"
                word_idx = self.vocab_index[word]
                emb_vector = self.model[word_idx]
                rtn_vector = emb_vector

        # final check before returning vector
        if DEBUG:
            print(">>> DEBUG: len(rtn_vector) = %s" % (len(rtn_vector)))
            input(">>> before returning vector ...")
        if len(rtn_vector) < 1:
            return np.zeros(self.vector_dim)
        else:
            if len(rtn_vector) == self.vector_dim:
                return rtn_vector
            # TODO: find a better way to represent unknown word by character to have same-size with word-vector-size
            # For now, I add 0 to the [current-len, expected-len]
            else:
                logging.debug("Model name = %s, Current word = %s, Current size = %s, expected size = %s"
                      %(self.name, word, len(rtn_vector), self.vector_dim))
                return np.append(rtn_vector, np.zeros(self.vector_dim - len(rtn_vector)))

    def get_vector_of_unknown(self, word):
        """
        If word is UNK, use char_vector model instead.
        :param word:
        :return:
        """
        # Here we handle features based on the w2v model where
        # numbers and punctuations are encoded as <punct>, <number>
        if self.name == model_constants.word2vec:
            # Only models trained with these placeholder tokens have them (e.g. the public
            # GoogleNews vectors do not); otherwise fall back to the character embedding.
            if self.is_number(word) and "<number>" in self.model:
                rtn_vector = self.model["<number>"]
            elif self.is_punct(word) and "<punct>" in self.model:
                rtn_vector = self.model["<punct>"]
            else:
                rtn_vector = self.get_char_vector(self.char_model, word)

            if rtn_vector is not None:
                if len(rtn_vector) > self.vector_dim:
                    logging.debug("auto trim to %s/%s dimensions", self.vector_dim, len(rtn_vector))
                    return rtn_vector[:self.vector_dim]
                else:
                    return rtn_vector
        # otherwise, using c2v to build-up the embedding vector
        else:
            return self.get_char_vector(self.char_model, word)


class embedding_models(object):
    """
    Using all available embedding models to generate vectors
    """
    def __init__(self, list_models):
        self.list_models = list_models # list of embedding_model_objs: ['word2vec', 'char2vec', 'private_word2vec']

    def add_model(self, emb_model, char_model):
        """
        Add new model into the collection of embedding models. Note that, every model has to add char_model to handle
        unknown word.
        :param emb_model:
        :param char_model:
        :return:
        """
        if char_model is None:
            raise Exception("char_model cannot be None.")
        if isinstance(emb_model, embedding_model):
            emb_model.set_char_model(char_model)
            self.list_models.append(emb_model)
        else:
            raise Exception("Not an instance of embedding_model class.")

    def get_vector_of_document(self, document):
        """
        Get all embedding vectors for one document
        :param document:
        :return:
        """
        doc_vector = []
        # debug_dict = {}
        # print ("len_doc = ", len(document))
        for word in document:

            all_vectors_of_word = []

            # get all embedding vectors of a word
            for emb_model in self.list_models:
                emb_vector = emb_model.get_word_vector(word)
                # print("len_emb_vector = ", len(emb_vector))
                all_vectors_of_word.extend(emb_vector)

                # if word in debug_dict.keys():
                #     debug_dict[word].append(len(emb_vector))
                # else:
                #     debug_dict[word] = [len(emb_vector)]

             # stack a combined vector of all words
            doc_vector.append(all_vectors_of_word)

        # print("list of words and emb size = ", debug_dict)
        # get the mean of them to represent a document
        doc_vector = np.mean(doc_vector, axis=0)

        return doc_vector

    def get_word_vector_of_multi_embeddings(self, word):
        """
        Get all embedding vectors for one document
        :param word:
        :return:
        """
        word_vector = []
        for emb_model in self.list_models:
            emb_vector = emb_model.get_word_vector(word)
            word_vector.extend(emb_vector)

        return word_vector