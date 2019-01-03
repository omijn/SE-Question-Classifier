""" Interactively make predictions using the classifier built in train.py """

from sklearn.externals import joblib
from preprocess import Preprocessor


class PredictionClient:
    def __init__(self, clf, preprocessor):
        self.clf = clf
        self.pp = preprocessor

    def get_user_input(self):
        return input("Enter a question to be classified: ")

    def predict(self, text):
        return self.pp.decode_labels(self.clf.predict(self.pp.vectorize_transform_text([text])))[0]


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
