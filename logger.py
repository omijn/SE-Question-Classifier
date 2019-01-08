import hashlib
import tensorflow as tf


def record_attempt(classifier, tokenizer, train_size, val_size, metrics, logfile='train_results'):
    classifier_name = classifier.__class__.__name__

    # keras model
    if isinstance(classifier, tf.keras.Sequential):
        config = classifier.get_config()
        config['layers'][0]['config'].pop('batch_input_shape')
        classifier_params = "Classifier params: " + str(config)
    # sklearn model
    else:
        classifier_params = "Classifier params: " + str(classifier.get_params())

    # keras tokenizer
    if isinstance(tokenizer, tf.keras.preprocessing.text.Tokenizer):
        config = tokenizer.get_config()
        [config.pop(key) for key in ['index_docs', 'index_word', 'word_counts', 'word_docs', 'word_index']]
        tokenizer_params = "Tfidf params: " + str(config)
    # sklearn tokenizer
    else:
        tokenizer_params = "Tfidf params: " + str(tokenizer.get_params())

    train_size = str(train_size)
    val_size = str(val_size)

    attempt_id = hashlib.sha256(
        bytes(classifier_name + classifier_params + tokenizer_params + train_size, encoding='utf-8')).hexdigest()

    train_score_msg = "{} score on training set = {}".format(metrics['train']['name'].capitalize(),
                                                             metrics['train']['value'])
    val_score_msg = "{} score on validation set = {}".format(metrics['val']['name'].capitalize(),
                                                             metrics['val']['value'])

    with open(logfile, "a") as f:
        f.write(attempt_id + "\n")
        f.write(classifier_name + ": " + train_size + "/" + val_size + "\n")
        f.write(classifier_params + "\n")
        f.write(tokenizer_params + "\n")
        f.write(train_score_msg + "\n")
        f.write(val_score_msg + "\n")
        f.write("----------------------------------------------\n\n")
