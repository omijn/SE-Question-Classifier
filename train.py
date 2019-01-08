""" Train a classifer on the question data obtained from scrape.py """

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import json
import numpy as np
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

from preprocess import DataManager, Preprocessor, label_encoder
import hashlib

from logger import record_attempt
import tensorflow as tf
from tensorflow.python.keras.layers import Dense, CuDNNLSTM, CuDNNGRU, LSTM, GRU, Dropout, SimpleRNN, Embedding, Input, Activation
from tensorflow.python.keras.utils import to_categorical
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.optimizers import RMSprop, Adam, Adagrad, SGD

from sklearn.utils import shuffle


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


# read data
def read_data(type='train'):
    data = {
        'X': [],
        'y': []
    }

    for category in ['pos', 'neg']:
        train_path = os.path.join('..', 'text_classification', 'aclImdb', type, category)
        review_files = os.listdir(train_path)
        for file in sorted(review_files):
            with open(os.path.join(train_path, file)) as f:
                data['X'].append(f.read())
        if category == 'pos':
            data['y'].extend([1] * len(review_files))
        else:
            data['y'].extend([0] * len(review_files))

    return shuffle(data['X'], data['y'])


def ngram_model(Xtrain, ytrain, Xval, yval):
    preprocessor = Preprocessor(vectorizer_mode='tfidf')

    # Xtrain = preprocessor.clean(Xtrain)
    # Xval = preprocessor.clean(Xval)

    # apply tf-idf vectorization on the text
    Xtrain = preprocessor.vectorize_fit_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    clf = MultinomialNB()
    clf.fit(Xtrain, ytrain)

    ytrain_pred = clf.predict(Xtrain)
    yval_pred = clf.predict(Xval)

    train_f1 = f1_score(ytrain, ytrain_pred, average='micro')
    val_f1 = f1_score(yval, yval_pred, average='micro')

    print("Train F1 score: {}".format(train_f1))
    print("Validation F1 score: {}".format(val_f1))

    record_attempt(
        classifier=clf,
        tokenizer=preprocessor.tfidf,
        train_size=len(ytrain),
        val_size=len(yval),
        metrics={
            'train': {
                'name': 'f1 (micro)',
                'value': train_f1
            },
            'val': {
                'name': 'f1 (micro)',
                'value': val_f1
            }
        }
    )

    return clf, preprocessor


def neural_ngram_model(Xtrain, ytrain, Xval, yval):
    preprocessor = Preprocessor(vectorizer_mode='tfidf')

    preprocessor.vectorize_fit_text(Xtrain)
    Xtrain = preprocessor.vectorize_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    num_classes = len(label_encoder.classes_)

    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)

    model = tf.keras.Sequential()
    model.add(Dense(300, activation='tanh', input_dim=Xtrain.shape[1]))
    model.add(Dropout(0.5))
    model.add(Dense(100, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, ytrain, epochs=3, batch_size=32, validation_data=(Xval, yval))

    train_score = model.evaluate(Xtrain, ytrain)[1]
    val_score = model.evaluate(Xval, yval)[1]

    record_attempt(
        classifier=model,
        tokenizer=preprocessor.tfidf,
        train_size=len(ytrain),
        val_size=len(yval),
        metrics={'train': {'name': 'accuracy', 'value': train_score}, 'val': {'name': 'accuracy', 'value': val_score}}
    )

    return model, preprocessor


def sequence_model(Xtrain, ytrain, Xval, yval):
    preprocessor = Preprocessor(vectorizer_mode='embeddings')

    Xtrain = preprocessor.vectorize_fit_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    num_classes = len(label_encoder.classes_)

    ytrain = to_categorical(ytrain, num_classes=num_classes)
    yval = to_categorical(yval, num_classes=num_classes)

    POST_LENGTH_LIMIT = 1500
    max_post_len = len(max(Xtrain, key=len))
    if max_post_len > POST_LENGTH_LIMIT:
        max_post_len = POST_LENGTH_LIMIT

    Xtrain = sequence.pad_sequences(Xtrain, max_post_len)
    Xval = sequence.pad_sequences(Xval, max_post_len)

    vocab_len = len(preprocessor.tokenizer.word_index.keys())

    # input = Input(shape=(max_post_len,))
    # embeddings = Embedding(input_dim=vocab_len + 1, output_dim=50, input_length=max_post_len)(input)
    # X = LSTM(128, return_sequences=True)(embeddings)
    # X = Dropout(0.5)(X)
    # X = LSTM(128, return_sequences=False)(embeddings)
    # X = Dropout(0.5)(X)
    # X = Dense(num_classes)(X)
    # X = Activation('softmax')(X)
    #
    # model = tf.keras.models.Model(inputs=input, outputs=X)

    checkpoint_path = "training_checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)

    model = tf.keras.Sequential()
    model.add(Embedding(
        input_dim=vocab_len + 1,
        output_dim=300,
        input_length=max_post_len
    ))
    model.add(LSTM(128, return_sequences=True))
    model.add(Dropout(0.25))
    model.add(LSTM(128, return_sequences=False))
    model.add(Dropout(0.25))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=RMSprop(lr=0.005), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, ytrain, batch_size=16, epochs=50, validation_data=(Xval, yval), callbacks=[cp_callback])

    train_score = model.evaluate(Xtrain, ytrain)[1]
    val_score = model.evaluate(Xval, yval)[1]

    print("Train accuracy: {}".format(train_score))
    print("Validation accuracy: {}".format(val_score))

    record_attempt(
        classifier=model,
        tokenizer=preprocessor.tokenizer,
        train_size=len(ytrain),
        val_size=len(yval),
        metrics={'train': {'name': 'accuracy', 'value': train_score}, 'val': {'name': 'accuracy', 'value': val_score}}
    )

    model.save(filepath='neural_seq_model.h5')
    joblib.dump(preprocessor, 'seq_preprocessor.sav')
    joblib.dump(label_encoder.classes_, 'label_encoder.sav')
    return model, preprocessor


def main():
    dm = DataManager(category_size=500, num_files=5)
    data = dm.read_data()

    # get sites in the form [<list of sites from qdata1.json>, <list of sites from qdata2.json>, ...]
    selected_sites = dm.get_all_keys(data)

    # from list of lists, create a flat list of the form: ["site1", "site2", "site3", ...]
    flat_labels = dm.flatten(selected_sites)

    # create mapping from string labels (site names) to numeric labels
    label_encoder.fit(flat_labels)

    # transform string labels into numeric labels using the mapping just created
    encoded_labels = label_encoder.transform(flat_labels)

    # for each site, get "dm.category_size" number of questions
    sample = dm.get_sample(data, selected_sites)

    # create dataset
    X, y = dm.create_dataset(sample, encoded_labels)

    # Google says find the samples/word ratio to determine which model to use
    # https://developers.google.com/machine-learning/guides/text-classification/step-2-5
    # print("S/W ratio = " + str(np.median([len(question.split()) for question in X])))
    # Xtrain, ytrain = read_data('train')
    # Xtest, ytest = read_data('test')
    # Xtrain, Xval, ytrain, yval = train_test_split(Xtrain, ytrain, train_size=0.8)

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.88)
    Xtest, Xval, ytest, yval = train_test_split(Xtest, ytest, train_size=0.5)

    model, preprocessor = sequence_model(Xtrain, ytrain, Xval, yval)
    pass

    # joblib.dump(clf, "classifier." + attempt_id + ".sav")
    # joblib.dump(preprocessor, "preprocessor." + attempt_id + ".sav")


if __name__ == '__main__':
    main()
