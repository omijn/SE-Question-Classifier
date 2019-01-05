from flask import Flask, render_template, request, jsonify
from preprocess import Preprocessor, DataManager
from predict import PredictionClient
from sklearn.externals import joblib
import json

app = Flask(__name__)

clf = joblib.load("classifier.sav")
pp = joblib.load("preprocessor.sav")
pc = PredictionClient(clf, pp)
site_metadata = json.load(open("sites.json"))['items']

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    data = request.get_json()
    question = data['question']
    prediction = pc.predict(question)
    return jsonify(prediction=prediction,
                   name=site_metadata[prediction]["name"],
                   url=site_metadata[prediction]["site_url"],
                   logo=site_metadata[prediction]["logo_url"]
                   )
