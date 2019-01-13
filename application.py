from flask import Flask, render_template, request, jsonify
from preprocess import Preprocessor, DataManager
from predict import PredictionClient
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder
import json
import tensorflow as tf
import time

app = Flask(__name__)

tf.keras.backend.clear_session()
default_model_name = "neural_ngram"
model = tf.keras.models.load_model("{}/best".format(default_model_name))
model._make_predict_function()
pp = joblib.load("{}/preprocessor_best".format(default_model_name))
le = LabelEncoder()
le.classes_ = joblib.load("{}/label_encoder_classes_best".format(default_model_name))
pc = PredictionClient(model, pp, le)
site_metadata = json.load(open("sites.json"))['items']


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=['POST', 'GET'])
def predict():
    data = request.get_json()
    question = data['question']
    top_n = str.lower(str.strip(data['top_n']))
    if top_n == "best":
        top_n = 1
    elif top_n == "top two":
        top_n = 2
    elif top_n == "top three":
        top_n = 3
    elif top_n == "top four":
        top_n = 4
    elif top_n == "top five":
        top_n = 5
    else:
        top_n = 1
    best_preds = pc.best_answers(question, top_n)
    return_obj = {"predictions": []}
    for pred in best_preds:
        return_obj["predictions"].append(
            {
                "prediction": pred,
                "name": site_metadata[pred]["name"],
                "url": site_metadata[pred]["site_url"],
                "logo": site_metadata[pred]["high_resolution_icon_url"]
            }
        )
    return jsonify(return_obj)
