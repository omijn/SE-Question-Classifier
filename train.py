import json
import pickle
import random
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer


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


def main():
    dm = DataManager(category_size=500, num_files=5)
    pp = Preprocessor()
    data = dm.read_data()

    # get sites in the form [<list of sites from qdata1.json>, <list of sites from qdata2.json>, ...]
    selected_sites = dm.get_all_keys(data)

    # from list of lists, create a flat list of the form: ["site1", "site2", "site3", ...]
    flat_labels = dm.flatten(selected_sites)

    # create mapping from string labels (site names) to numeric labels
    pp.fit_labelencoder(flat_labels)

    # transform string labels into numeric labels using the mapping just created
    encoded_labels = pp.encode_labels(flat_labels)

    # for each site, get "dm.category_size" number of questions
    sample = dm.get_sample(data, selected_sites)

    # create dataset
    X, y = dm.create_dataset(sample, encoded_labels)

    # the data is now ready for use
    ### INSERT MACHINE LEARNING HERE

    # apply tf-idf vectorization on the text
    X = pp.vectorize_fit_transform_text(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    # X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # save model
    pickle.dump(clf, open("classifier.sav", "wb"))
    pickle.dump(pp, open("preprocessor.sav", "wb"))
    # pp.save()

    # y_pred_val = clf.predict(X_val)
    y_pred_test = clf.predict(X_test)
    y_pred_all_data = clf.predict(X)

    print("Main classifier accuracy over the entire dataset = {}".format(accuracy_score(y_pred_all_data, y)))
    print("Main classifier F1 score over the entire dataset = {}".format(f1_score(y_pred_all_data, y, average='macro')))
    print("Main classifier F1 score over the entire dataset = {}".format(f1_score(y_pred_all_data, y, average='micro')))

    # print("Validation set result")
    # print(classification_report(y_val, y_pred_val, target_names=dm.flatten(selected_sites)))

    print("Test set result")
    print(classification_report(y_test, y_pred_test, target_names=flat_labels))


if __name__ == '__main__':
    main()
