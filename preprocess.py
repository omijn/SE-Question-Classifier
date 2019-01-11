import json
import random
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.python.keras.preprocessing import text, sequence
import re
from nltk.stem import SnowballStemmer

np.random.seed(10)


class DataManager:

    def __init__(self):
        self.data = {}
        self.labels = []

    def read_data_from_files(self, num_files=6, excluded_sites=[]):
        file_handles = [open("qdata" + str(f) + ".json") for f in range(1, num_files + 1)]
        self.data = {site: data for f in file_handles for site, data in (json.load(f)).items() if site not in excluded_sites}
        self.labels = list(self.data.keys())

    def get_distinct_labels(self):
        return self.labels

    def get_xy(self, questions_per_site=1000):
        X = []
        y = []
        for site, qdata_list in self.data.items():
            randomly_sampled_qdata_list = np.random.permutation(qdata_list)[:questions_per_site]
            num_qdata = len(randomly_sampled_qdata_list)
            for qdata in randomly_sampled_qdata_list:
                X.append(list(qdata.values())[0]['content'])
            y.extend([site] * num_qdata)
        return X, y

    def train_test_val_split(self, X, y, train_size, test_size, val_size):
        data_after_train_split = 1 - train_size
        test_prop = test_size / data_after_train_split
        val_prop = 1 - test_prop
        Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=train_size)
        Xtest, Xval, ytest, yval = train_test_split(Xtest, ytest, train_size=test_prop)
        return Xtrain, Xtest, Xval, ytrain, ytest, yval

    def single_sample(self, X, y, tfidf, index):
        print(np.array(sorted(tfidf.inverse_transform(tfidf.transform([X[index]]))[0])))
        print(X[index])
        print(label_encoder.inverse_transform([y[index]]))


TFIDF_MODE = 0
EMBEDDING_MODE = 1


class Preprocessor:
    def __init__(self, vectorizer_mode=TFIDF_MODE, max_features=45000, verbose=False):
        self.le = LabelEncoder()
        self.stemmer = SnowballStemmer("english")
        self.vectorizer_mode = vectorizer_mode
        self.max_features = max_features
        self.verbose = verbose
        if vectorizer_mode == TFIDF_MODE:
            # self.stopwords = ['the', 'a', 'in', 'of', 'that', 'which', 'what', 'or', 'and']
            self.tfidf = TfidfVectorizer(stop_words='english', max_features=self.max_features, ngram_range=(1, 2))
        elif vectorizer_mode == EMBEDDING_MODE:
            self.tokenizer = text.Tokenizer(num_words=self.max_features)

    def fit_labelencoder(self, text_labels):
        self.le.fit(text_labels)

    def encode_labels(self, text_labels):
        return self.le.transform(text_labels)

    def decode_labels(self, numeric_labels):
        return self.le.inverse_transform(numeric_labels)

    def clean(self, data, remove_http_urls=True):
        if remove_http_urls:
            url_pattern = re.compile(r'(?:https?://)(?:\w+\.)+(?:com|org|gov|edu|uk|net|us|co|info|ly).*?(?=\s)',
                                     flags=re.IGNORECASE)
            for i in range(len(data)):
                data[i] = url_pattern.sub("", data[i])
        return data

    def stem(self, textdata):
        self.verbose and print("Stemming {} texts.".format(len(textdata)))
        for i in range(len(textdata)):
            textdata[i] = self.stemmer.stem(textdata[i])
        return textdata

    def vectorize_fit_text(self, textdata):
        if self.vectorizer_mode == TFIDF_MODE:
            self.verbose and print("Fitting {} texts to Tfidfvectorizer".format(len(textdata)))
            self.tfidf.fit(textdata)
        elif self.vectorizer_mode == EMBEDDING_MODE:
            self.verbose and print("Fitting {} texts to Tokenizer".format(len(textdata)))
            self.tokenizer.fit_on_texts(textdata)

    def vectorize_fit_transform_text(self, textdata):
        self.vectorize_fit_text(textdata)
        return self.vectorize_transform_text(textdata)

    def vectorize_transform_text(self, textdata):
        if self.vectorizer_mode == TFIDF_MODE:
            self.verbose and print("Transforming {} texts to Tf-Idf vectors".format(len(textdata)))
            return self.tfidf.transform(textdata)
        elif self.vectorizer_mode == EMBEDDING_MODE:
            self.verbose and print("Transforming {} texts to sequences".format(len(textdata)))
            return self.tokenizer.texts_to_sequences(textdata)

    def get_tokenizer(self):
        if self.vectorizer_mode == TFIDF_MODE:
            return self.tfidf
        elif self.vectorizer_mode == EMBEDDING_MODE:
            return self.tokenizer
    # def save(self):
    #     np.save('labelencoder_classes.npy', self.le.classes_)
    #     pickle.dump(self.tfidf, open("tfidf.sav", "wb"))


label_encoder = LabelEncoder()
