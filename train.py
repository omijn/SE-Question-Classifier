""" Train a classifer on the question data obtained from scrape.py """

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import h5py
import gensim
import numpy as np
import tensorflow as tf
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from tensorflow.python.keras.layers import Dense, LSTM, CuDNNLSTM, Dropout, Embedding, SeparableConv1D, MaxPooling1D, \
    GlobalAveragePooling1D, LeakyReLU
from tensorflow.python.keras.optimizers import RMSprop, Adam
from tensorflow.python.keras.preprocessing import sequence
from tensorflow.python.keras.utils import to_categorical

from logger import record_attempt
from preprocess import DataManager, Preprocessor, label_encoder, TFIDF_MODE, EMBEDDING_MODE

DEBUG = 0
num_classes = 0
TENSORBOARD_LOGDIR = "tensorboard_logdir"

def error_analysis(yval, yval_pred, Xval, Xval_original, preprocessor):
    original_words = Xval_original
    selected_words = preprocessor.tfidf.inverse_transform(Xval)
    incorrect_predictions_indices = np.argwhere(yval_pred != yval)
    predicted_labels = label_encoder.inverse_transform(yval_pred)
    actual_labels = label_encoder.inverse_transform(yval)
    for index in incorrect_predictions_indices.reshape(-1):
        print("\nOriginal X: {}\n\nSelected words from X: {}\n\nPredicted {}\nActually {}".format(original_words[index],
                                                                                                  selected_words[index],
                                                                                                  predicted_labels[
                                                                                                      index],
                                                                                                  actual_labels[index]))
        i = input("Press Enter to view next")
        if i == "quit":
            break


def ngram_model(Xtrain, ytrain, Xval, yval):
    model_name = "mnb_ngram"
    preprocessor = Preprocessor(vectorizer_mode=TFIDF_MODE, max_features=45000, verbose=True)

    # apply tf-idf vectorization on the text
    Xtrain = preprocessor.vectorize_fit_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    # define classifier and train
    model = MultinomialNB(alpha=0.4)
    model.fit(Xtrain, ytrain)

    # make predictions
    ytrain_pred = model.predict(Xtrain)
    yval_pred = model.predict(Xval)

    prediction_probs = model.predict_proba(Xval).argsort(axis=1)
    best_preds = prediction_probs[:, -1]
    second_best_preds = prediction_probs[:, -2]

    label_in_top_two_preds = np.full((len(yval),), True)
    for i in range(len(yval)):
        if yval[i] != best_preds[i] and yval[i] != second_best_preds[i]:
            label_in_top_two_preds[i] = False

    val_acc_top_two = label_in_top_two_preds.sum() / len(yval)

    train_acc = accuracy_score(ytrain, ytrain_pred)
    val_acc = accuracy_score(yval, yval_pred)

    print("Validation accuracy (label in top 2 predictions): {}".format(val_acc_top_two))

    return model, model_name, preprocessor, ytrain_pred, yval_pred, "accuracy", train_acc, val_acc


def neural_ngram_model(Xtrain, ytrain, Xval, yval):
    model_name = "neural_ngram"
    os.makedirs(model_name, exist_ok=True)
    tensorboard_logdir = os.path.join(model_name, TENSORBOARD_LOGDIR)
    os.makedirs(tensorboard_logdir, exist_ok=True)
    preprocessor = Preprocessor(vectorizer_mode=TFIDF_MODE)

    # apply tf-idf vectorization on the text
    Xtrain = preprocessor.vectorize_fit_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    # Converting labels into one hot vectors
    ytrain = to_categorical(ytrain, num_classes)
    yval = to_categorical(yval, num_classes)

    checkpoint_path = "{}/cp.ckpt".format(model_name)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, write_graph=False, histogram_freq=1, write_grads=True, batch_size=64)

    model = tf.keras.Sequential()
    model.add(Dense(500, input_dim=Xtrain.shape[1]))
    model.add(LeakyReLU())
    model.add(Dropout(0.5))
    # model.add(Dense(1000, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, ytrain, epochs=5, batch_size=8, validation_data=(Xval, yval), callbacks=[checkpoint_callback, earlystopping_callback, tensorboard_callback])

    train_acc = model.evaluate(Xtrain, ytrain)[1]
    val_acc = model.evaluate(Xval, yval)[1]

    ytrain_pred = model.predict_classes(Xtrain)
    yval_pred = model.predict_classes(Xval)

    return model, model_name, preprocessor, ytrain_pred, yval_pred, "accuracy", train_acc, val_acc


def create_pretrained_embedding_layer(tokenizer, input_length):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    embedding_dimension = len(word2vec_model['porcupine'])

    vocab_len = len(tokenizer.word_index.keys()) + 1

    embedding_matrix = np.zeros((tokenizer.num_words, embedding_dimension))

    for word, index in list(tokenizer.word_index.items())[:tokenizer.num_words - 1]:
        try:
            embedding_matrix[index, :] = word2vec_model[word]
        except KeyError as e:
            continue

    embedding_layer = Embedding(input_dim=tokenizer.num_words, output_dim=embedding_dimension,
                                input_length=input_length, weights=[embedding_matrix], trainable=True)

    return embedding_layer, embedding_matrix


def tfidf_weighted_embeddings_neural_ngram_model(Xtrain, ytrain, Xval, yval):
    word2vec_model = gensim.models.KeyedVectors.load_word2vec_format("GoogleNews-vectors-negative300.bin", binary=True)
    embedding_dimension = len(word2vec_model['porcupine'])

    preprocessor = Preprocessor(TFIDF_MODE)
    Xtrain = preprocessor.vectorize_fit_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    num_classes = len(label_encoder.classes_)

    ytrain = to_categorical(ytrain, num_classes=num_classes)
    yval = to_categorical(yval, num_classes=num_classes)

    vocabulary = preprocessor.tfidf.vocabulary_
    vocab_size = len(vocabulary)

    # construct embedding matrix E (vocab_size, embedding_dimension)
    oov_embedding = np.zeros(embedding_dimension)
    E = np.zeros((vocab_size, embedding_dimension))
    for word, index in vocabulary.items():
        try:
            E[index, :] = word2vec_model[word]
        except KeyError as e:
            E[index, :] = oov_embedding

    Xtrain_weighted_avg_embeddings = Xtrain.dot(E) / np.sum(Xtrain, axis=1)
    Xval_weighted_avg_embeddings = Xval.dot(E) / np.sum(Xval, axis=1)

    model = tf.keras.Sequential()
    model.add(Dense(200, activation='tanh', input_dim=embedding_dimension))
    model.add(Dense(200, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(Xtrain_weighted_avg_embeddings, ytrain, batch_size=32, epochs=500,
              validation_data=(Xval_weighted_avg_embeddings, yval))

    model.evaluate(Xtrain_weighted_avg_embeddings, ytrain)
    model.evaluate(Xval_weighted_avg_embeddings, yval)

    return model, preprocessor


def sequence_model(Xtrain, ytrain, Xval, yval):
    model_name = "recurrent_lstm_embedding"
    tensorboard_logdir = os.path.join(model_name, TENSORBOARD_LOGDIR)
    os.makedirs(model_name, exist_ok=True)
    os.makedirs(tensorboard_logdir, exist_ok=True)
    preprocessor = Preprocessor(vectorizer_mode=EMBEDDING_MODE)

    Xtrain = preprocessor.vectorize_fit_transform_text(Xtrain)
    Xval = preprocessor.vectorize_transform_text(Xval)

    ytrain = to_categorical(ytrain, num_classes=num_classes)
    yval = to_categorical(yval, num_classes=num_classes)

    POST_LENGTH_LIMIT = 500
    max_post_len = len(max(Xtrain, key=len))
    if max_post_len > POST_LENGTH_LIMIT:
        max_post_len = POST_LENGTH_LIMIT

    Xtrain = sequence.pad_sequences(Xtrain, max_post_len)
    Xval = sequence.pad_sequences(Xval, max_post_len)

    embedding_layer, emb_matrix = create_pretrained_embedding_layer(preprocessor.tokenizer, max_post_len)

    checkpoint_path = "{}/cp.ckpt".format(model_name)
    checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, save_best_only=True, verbose=1)
    earlystopping_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=tensorboard_logdir, histogram_freq=3, write_graph=False,
                                                          write_grads=False, write_images=False)


    model = tf.keras.Sequential()
    model.add(embedding_layer)
    model.add(LSTM(256, return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(256, return_sequences=False))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    model.compile(optimizer=RMSprop(lr=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, ytrain, batch_size=16, epochs=50, validation_data=(Xval, yval), callbacks=[checkpoint_callback, earlystopping_callback, tensorboard_callback])

    train_acc = model.evaluate(Xtrain, ytrain)[1]
    val_acc = model.evaluate(Xval, yval)[1]

    ytrain_pred = model.predict_classes(Xtrain)
    yval_pred = model.predict_classes(Xval)

    return model, model_name, preprocessor, ytrain_pred, yval_pred, "accuracy", train_acc, val_acc


def sepCNN(Xtrain, ytrain, Xval, yval):
    preprocessor = Preprocessor(vectorizer_mode=EMBEDDING_MODE)

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

    embedding_layer = create_pretrained_embedding_layer(preprocessor.tokenizer, max_post_len)

    checkpoint_path = "training_checkpoints/cp.ckpt"
    checkpoint_dir = os.path.dirname(checkpoint_path)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path, verbose=1)

    model = tf.keras.Sequential()
    model.add(embedding_layer)
    for _ in range(3):
        model.add(Dropout(0.4))
        model.add(SeparableConv1D(128, 3, padding='same', activation='relu', bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform'))
        model.add(SeparableConv1D(128, 3, padding='same', activation='relu', bias_initializer='random_uniform',
                                  depthwise_initializer='random_uniform'))
        model.add(MaxPooling1D())

    model.add(SeparableConv1D(256, 3, padding='same', activation='relu', bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform'))
    model.add(SeparableConv1D(256, 3, padding='same', activation='relu', bias_initializer='random_uniform',
                              depthwise_initializer='random_uniform'))
    model.add(GlobalAveragePooling1D())
    model.add(Dropout(0.4))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(lr=0.003), loss='categorical_crossentropy', metrics=['accuracy'])

    model.fit(Xtrain, ytrain, batch_size=16, epochs=200, validation_data=(Xval, yval), callbacks=[cp_callback])

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

    model.save(filepath='sepCNN.h5')
    joblib.dump(preprocessor, 'seq_preprocessor.sav')
    joblib.dump(label_encoder.classes_, 'label_encoder.sav')
    return model, preprocessor


def main():
    dm = DataManager(category_size=450, num_files=5)
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

    global num_classes
    num_classes = len(label_encoder.classes_)

    # Google says find the samples/word ratio to determine which model to use
    # https://developers.google.com/machine-learning/guides/text-classification/step-2-5
    # print("S/W ratio = " + str(np.median([len(question.split()) for question in X])))

    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, train_size=0.88)
    Xtest, Xval, ytest, yval = train_test_split(Xtest, ytest, train_size=0.5)

    model, model_name, preprocessor, ytrain_pred, yval_pred, metric_name, train_score, val_score = sequence_model(Xtrain,
                                                                                                               ytrain,
                                                                                                               Xval,
                                                                                                               yval)

    print("Train {} score: {}".format(metric_name, train_score))
    print("Validation {} score: {}".format(metric_name, val_score))

    record_attempt(classifier=model, tokenizer=preprocessor.get_tokenizer(), train_size=len(ytrain), val_size=len(yval),
                   metrics={
                       'train': {
                           'name': metric_name,
                           'value': train_score
                       },
                       'val': {
                           'name': metric_name,
                           'value': val_score
                       }
                   }
                   )

    # error_analysis(yval, yval_pred, Xval, Xval_original, preprocessor)

    os.makedirs(model_name, exist_ok=True)
    if isinstance(model, tf.keras.Sequential):
        model.save("{}/model.h5".format(model_name))
    else:
        joblib.dump(model, "{}/model.sav".format(model_name))

    joblib.dump(preprocessor, "{}/preprocessor.sav".format(model_name))
    joblib.dump(label_encoder.classes_, "{}/label_encoder_classes.sav".format(model_name))


if __name__ == '__main__':
    main()
