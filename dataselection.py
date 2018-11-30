import json
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score


class DataManager:

    def __init__(self, category_size=100):
        self.category_size = category_size

    def read_data(self):
        file_handles = [open("qdata" + str(f) + ".json") for f in range(5)]
        self.data = [json.load(f) for f in file_handles]
        return self.data

    def get_keys(self, data):
        return [d.keys() for d in data]

    def get_sample(self, data, site_lists):
        return [[data[i][site][:self.category_size] for site in site_list] for i, site_list in enumerate(site_lists)]

    def flatten(self, lol):
        return [item for sublist in lol for item in sublist]

    def get_flat_keys(self):
        keys = self.get_keys()
        return [site for lst in keys for site in lst]

    def create_data(self, sample, labels):
        pp = Preprocessor()
        flat_labels = self.flatten(labels)
        pp.fit_labels(flat_labels)
        encoded_labels = pp.encode_labels(flat_labels)
        flat_X = self.flatten(sample)
        moreflat_X = self.flatten(flat_X)
        titles_X = [item[0]['title'] for item in (list(d.values()) for d in moreflat_X)]
        flat_y = [val for val in encoded_labels for _ in range(self.category_size)]
        return titles_X, flat_y


class Preprocessor:
    def __init__(self):
        self.le = LabelEncoder()
        self.tfidf = TfidfVectorizer()

    def fit_labels(self, text_labels):
        self.le.fit(text_labels)

    def encode_labels(self, text_labels):
        return self.le.transform(text_labels)

    def decode_labels(self, numeric_labels):
        return self.le.inverse_transform(numeric_labels)

    def vectorize_text(self, textdata):
        return self.tfidf.fit_transform(textdata)




def main():
    dm = DataManager()
    pp = Preprocessor()
    data = dm.read_data()
    site_lists = [['stackoverflow', 'apple'], ['sports', 'politics'], ['astronomy', 'aviation'],
                  ['mythology', 'lifehacks'], ['literature', 'vegetarianism']]
    sample = dm.get_sample(data, site_lists)
    X, y = dm.create_data(sample, site_lists)
    X = pp.vectorize_text(X)

    X_train, X_test, y_train, y_test = train_test_split(X, y)

    clf = MultinomialNB()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f1_score(y_pred, y_test, average='micro'))

    pass


if __name__ == '__main__':
    main()
