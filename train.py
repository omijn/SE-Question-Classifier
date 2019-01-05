""" Train a classifer on the question data obtained from scrape.py """

import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier

from preprocess import DataManager, Preprocessor


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

    # Google says find the samples/word ratio to determine which model to use
    # https://developers.google.com/machine-learning/guides/text-classification/step-2-5
    # print("S/W ratio = " + str(np.median([len(question.split()) for question in X])))

    # the data is now ready for use
    ### INSERT MACHINE LEARNING HERE

    # apply tf-idf vectorization on the text
    X = pp.vectorize_fit_transform_text(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)

    # save model
    # joblib.dump(clf, "classifier.sav")
    # joblib.dump(pp, "preprocessor.sav")
    # pp.save()

    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)

    algorithm = clf.__class__.__name__
    f1_train = f1_score(y_train, y_pred_train, average='micro')
    f1_val = f1_score(y_val, y_pred_val, average='micro')

    f1_train_msg = "F1 score (micro) on training set = {}".format(f1_train)
    f1_val_msg = "F1 score (micro) on validation set = {}".format(f1_val)

    print(f1_train_msg)
    print(f1_val_msg)
    with open("train_results", "a") as f:
        f.write(algorithm + "\n")
        f.write(f1_train_msg + "\n")
        f.write(f1_val_msg + "\n")
        f.write("------------------------------\n\n")

    # print("Validation set result")
    # print(classification_report(y_val, y_pred_val, target_names=dm.flatten(selected_sites)))


if __name__ == '__main__':
    main()
