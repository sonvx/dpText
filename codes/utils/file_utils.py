import pickle


def save_obj(obj, file_path):
    with open(file_path + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(file_path):
    with open(file_path + '.pkl', 'rb') as f:
        return pickle.load(f)
