""" Train a classifer on the question data obtained from scrape.py """

import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from preprocess import DataManager, Preprocessor
import hashlib

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.88)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, train_size=0.5)

    clf = MultinomialNB(alpha=0.4)
    clf = SVC(gamma='auto')
    clf = GridSearchCV(clf, {
        'kernel': ['rbf', 'poly'],
        'C': [0.01, 0.1, 1, 10, 100]
    })

    # clf = MLPClassifier()
    clf.fit(X_train, y_train)

    y_pred_train = clf.predict(X_train)
    y_pred_val = clf.predict(X_val)

    classifier_name = clf.__class__.__name__
    classifier_params = "Classifier params: " + str(clf.get_params())
    tfidf_params = "Tfidf params: " + str(pp.tfidf.get_params())
    train_size = str(len(y_train))
    val_size = str(len(y_val))

    attempt_id = hashlib.sha256(bytes(classifier_name + classifier_params + tfidf_params + train_size, encoding='utf-8')).hexdigest()

    # len(y_test)
    f1_train = f1_score(y_train, y_pred_train, average='micro')
    f1_val = f1_score(y_val, y_pred_val, average='micro')
    # acc_train = accuracy_score(y_train, y_pred_train)
    # acc_val = accuracy_score(y_val, y_pred_val)

    f1_train_msg = "F1 score (micro) on training set = {}".format(f1_train)
    f1_val_msg = "F1 score (micro) on validation set = {}".format(f1_val)
    # acc_train_msg = "Accuracy on training set = {}".format(acc_train)
    # acc_val_msg = "Accuracy on validation set = {}".format(acc_val)

    print(f1_train_msg)
    print(f1_val_msg)
    print(clf.cv_results_)
    print(clf.best_params_)
    # print(acc_train_msg)
    # print(acc_val_msg)

    with open("train_results", "a") as f:
        f.write(attempt_id + "\n")
        f.write(classifier_name + ": " + train_size + "/" + val_size + "\n")
        f.write(classifier_params + "\n")
        f.write(tfidf_params + "\n")
        f.write(f1_train_msg + "\n")
        f.write(f1_val_msg + "\n")
        # f.write(acc_train_msg + "\n")
        # f.write(acc_val_msg + "\n")
        f.write("----------------------------------------------\n\n")

    joblib.dump(clf, "classifier." + attempt_id + ".sav")
    joblib.dump(pp, "preprocessor." + attempt_id + ".sav")

    # print("Validation set result")
    # print(classification_report(y_val, y_pred_val, target_names=dm.flatten(selected_sites)))


if __name__ == '__main__':
    main()
