import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from tensorflow.python.keras.preprocessing import text, sequence
import re

# np.random.seed(0)


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
        return [[np.random.permutation(data[i][site])[:self.category_size] for site in site_list] for i, site_list in
                enumerate(site_lists)]

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

    def single_sample(self, X, y, tfidf, index):
        print(np.array(sorted(tfidf.inverse_transform(tfidf.transform([X[index]]))[0])))
        print(X[index])
        print(label_encoder.inverse_transform([y[index]]))


class Preprocessor:
    def __init__(self, vectorizer_mode='tfidf'):
        self.le = LabelEncoder()
        self.vectorizer_mode = vectorizer_mode

        if vectorizer_mode == 'tfidf':
            self.stopwords = ['the', 'a', 'in', 'of', 'that', 'which', 'what', 'or', 'and']
            self.tfidf = TfidfVectorizer(stop_words='english', max_features=60000, ngram_range=(1, 3))
        elif vectorizer_mode == 'embeddings':
            self.tokenizer = text.Tokenizer(num_words=40000)

    def fit_labelencoder(self, text_labels):
        self.le.fit(text_labels)

    def encode_labels(self, text_labels):
        return self.le.transform(text_labels)

    def decode_labels(self, numeric_labels):
        return self.le.inverse_transform(numeric_labels)

    def clean(self, data, remove_http_urls=True):
        if remove_http_urls:
            url_pattern = re.compile(r'(?:https?://)(?:\w+\.)+(?:com|org|gov|edu|uk|net|us|co|info|ly).*?(?=\s)', flags=re.IGNORECASE)
            for i in range(len(data)):
                data[i] = url_pattern.sub("", data[i])
        return data

    def vectorize_fit_text(self, textdata):
        if self.vectorizer_mode == 'tfidf':
            self.tfidf.fit(textdata)
        elif self.vectorizer_mode == 'embeddings':
            self.tokenizer.fit_on_texts(textdata)

    def vectorize_fit_transform_text(self, textdata):
        self.vectorize_fit_text(textdata)
        return self.vectorize_transform_text(textdata)

    def vectorize_transform_text(self, textdata):
        if self.vectorizer_mode == 'tfidf':
            return self.tfidf.transform(textdata)
        elif self.vectorizer_mode == 'embeddings':
            return self.tokenizer.texts_to_sequences(textdata)

    # def save(self):
    #     np.save('labelencoder_classes.npy', self.le.classes_)
    #     pickle.dump(self.tfidf, open("tfidf.sav", "wb"))

label_encoder = LabelEncoder()