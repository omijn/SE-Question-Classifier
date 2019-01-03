import json
import random

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder


class DataManager:

    def __init__(self, category_size=100, num_files=6):
        self.category_size = category_size
        self.num_files = num_files
        self.data = []

    def read_data(self):
        file_handles = [open("qdata" + str(f) + ".json") for f in range(1, self.num_files + 1)]
        self.data = [json.load(f) for f in file_handles]
        self.data[0].pop('meta.superuser')
        self.data[0].pop('meta.serverfault')
        return self.data

    def get_random_keys(self, num_categories_per_chunk, data):
        all_keys = self.get_all_keys(data)
        return [random.sample(keylist, num_categories_per_chunk) for keylist in all_keys]

    def get_all_keys(self, data):
        return [d.keys() for d in data]

    def get_sample(self, data, site_lists):
        return [[data[i][site][:self.category_size] for site in site_list] for i, site_list in enumerate(site_lists)]

    def flatten(self, lol):
        return [item for sublist in lol for item in sublist]

    def get_flat_keys(self):
        keys = self.get_all_keys()
        return [site for lst in keys for site in lst]

    def create_dataset(self, sample, encoded_labels):
        flat_X = self.flatten(sample)
        class_sizes = [len(flat_X[i]) for i in range(len(flat_X))]
        moreflat_X = self.flatten(flat_X)
        titles_X = [item[0]['content'] for item in (list(d.values()) for d in moreflat_X)]
        flat_y = self.flatten([label] * repeats for label, repeats in zip(encoded_labels, class_sizes))
        return titles_X, flat_y


class Preprocessor:
    def __init__(self):
        self.le = LabelEncoder()
        self.tfidf = TfidfVectorizer(stop_words='english')

    def fit_labelencoder(self, text_labels):
        self.le.fit(text_labels)

    def encode_labels(self, text_labels):
        return self.le.transform(text_labels)

    def decode_labels(self, numeric_labels):
        return self.le.inverse_transform(numeric_labels)

    def vectorize_fit_transform_text(self, textdata):
        return self.tfidf.fit_transform(textdata)

    def vectorize_transform_text(self, textdata):
        return self.tfidf.transform(textdata)

    # def save(self):
    #     np.save('labelencoder_classes.npy', self.le.classes_)
    #     pickle.dump(self.tfidf, open("tfidf.sav", "wb"))
