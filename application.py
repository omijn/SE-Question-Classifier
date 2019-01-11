from flask import Flask, render_template, request, jsonify
from preprocess import Preprocessor, DataManager
from predict import PredictionClient
from sklearn.externals import joblib
import json
import tensorflow as tf

app = Flask(__name__)

# tf.keras.backend.clear_session()
default_model = "neural_ngram"
model = tf.keras.models.load_model("{}/model.h5".format(default_model))
pp = joblib.load("{}/preprocessor.sav".format(default_model))
le = joblib.load("{}/label_encoder_classes.sav".format(default_model))
pc = PredictionClient(model, pp, le)
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
