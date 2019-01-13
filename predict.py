""" Interactively make predictions using the classifier built in train.py """

from sklearn.externals import joblib
from preprocess import Preprocessor, custom_tokenizer, compiled_url_pattern, compiled_digit_pattern, compiled_token_pattern, lemmatizer
import tensorflow as tf

tf.keras.backend.clear_session()
class PredictionClient:
    def __init__(self, model, preprocessor, label_encoder):
        self.model = model
        if isinstance(self.model, tf.keras.Sequential):
            self.model_type = "tensorflow"
        else:
            self.model_type = "sklearn"
        self.pp = preprocessor
        self.le = label_encoder

    def get_user_input(self):
        return input("Enter a question to be classified: ")

    def best_answers(self, x, n=1):
        # x = self._stem([x])
        x = self._transform_text(x)
        return self.le.inverse_transform(self.model.predict_proba([x])[0].argsort()[::-1][:n])

    def _stem(self, text):
        return self.pp.stem(text)

    def _transform_text(self, text):
        if isinstance(text, str):
            text = [text]
        return self.pp.vectorize_transform_text(text)


def main():
    clf = joblib.load("classifier.sav")
    pp = joblib.load("preprocessor.sav")
    pc = PredictionClient(clf, pp)

    question = ""
    while question != "quit":
        question = pc.get_user_input()
        if str.lower(question) == "quit":
            break
        print("Your question most likely belongs on the \"" + pc.predict(question) + "\" Stack Exchange site.\n")


if __name__ == '__main__':
    main()
