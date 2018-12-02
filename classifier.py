import json
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer


class DataManager:

    def __init__(self, category_size=100):
        self.category_size = category_size

    def read_data(self):
        file_handles = [open("qdata" + str(f) + ".json") for f in range(5)]
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


class PredictionClient:

    def __init__(self, clf, preprocessor):
        self.clf = clf
        self.pp = preprocessor

    def get_user_input(self):
        return input("Enter a question to be classified: ")

    def predict(self, text):
        return self.pp.decode_labels(self.clf.predict(self.pp.vectorize_transform_text([text])))[0]


def baseline(distinct_labels, pp, X, y):
    # create a vocabulary for each label
    tagfile = open("tags.json")
    tagdata = json.load(tagfile)
    vocabs = [tagdata['items'][label] for label in distinct_labels]

    # create a CountVectorizer for each label
    cvs = [CountVectorizer(vocabulary=vocab) for vocab in vocabs]

    # run each CountVectorizer on the corpus
    countVectorizedData = [cv.fit_transform(X).toarray().sum(axis=1) for cv in cvs]
    y_pred = pp.encode_labels(np.array(distinct_labels)[np.argmax(np.vstack(countVectorizedData), axis=0)])
    print("Baseline classifier accuracy over entire dataset = {}".format(accuracy_score(y, y_pred)))
    print("Baseline classifier F1 score over entire dataset = {}".format(f1_score(y, y_pred, average='macro')))
    print("Baseline classifier F1 score over entire dataset = {}".format(f1_score(y, y_pred, average='micro')))


def create_json(X, y):
    print(type(X))
    print(type(y))
    d = {
        "description": [
            "Body of questions posted on Stack Exchange websites, labelled as the name of the website they were posted on"],
        "authors": {
            "author1": "Abhijit Kashyap"
        },
        "emails": {
            "email1": "abhijitk@usc.edu"
        }
    }

    d['corpus'] = [{"label": int(example[1]), "data": example[0]} for example in zip(X, y)]
    f = open("final.json", "w")
    json.dump(d, f)


def main():
    dm = DataManager(500)
    pp = Preprocessor()
    data = dm.read_data()
    # selected_sites = [['stackoverflow', 'apple'], ['sports', 'politics'], ['astronomy', 'aviation'], ['mythology', 'lifehacks'], ['literature', 'vegetarianism']]
    selected_sites = dm.get_all_keys(data)
    # selected_sites = dm.get_random_keys(10, data)

    flat_labels = dm.flatten(selected_sites)
    pp.fit_labelencoder(flat_labels)
    encoded_labels = pp.encode_labels(flat_labels)

    sample = dm.get_sample(data, selected_sites)
    X, y = dm.create_dataset(sample, encoded_labels)

    create_json(X, y)

    # baseline(flat_labels, pp, X, y)

    X = pp.vectorize_fit_transform_text(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    # y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
    y_pred_all_data = clf.predict(X)

    print("Main classifier accuracy over the entire dataset = {}".format(accuracy_score(y_pred_all_data, y)))
    print("Main classifier F1 score over the entire dataset = {}".format(f1_score(y_pred_all_data, y, average='macro')))
    print("Main classifier F1 score over the entire dataset = {}".format(f1_score(y_pred_all_data, y, average='micro')))

    # print("Validation set result")
    # print(classification_report(y_val, y_pred_val, target_names=dm.flatten(selected_sites)))

    print("Test set result")
    print(classification_report(y_test, y_pred_test, target_names=dm.flatten(selected_sites)))

    pc = PredictionClient(clf, pp)

    question = ""
    while question != "quit":
        question = pc.get_user_input()
        if str.lower(question) == "quit":
            break
        print("Your question most likely belongs on the \"" + pc.predict(question) + "\" StackExchange site.\n")
    pass


if __name__ == '__main__':
    main()
