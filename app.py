from flask import Flask, render_template, request
from preprocessing import Preprocess
import joblib
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from word_tokenize_wrapper_function import CustomTokenizer
preprocess = Preprocess()

app = Flask(__name__)

model = joblib.load('model/sentiment_model_maps (1).pkl')


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/predict", methods=['POST'])
def predict():
    query_asis = [str(x) for x in request.form.values()]
    query_processed_list = [preprocess.stemming(preprocess.filtering(preprocess.normalization(
        preprocess.cleansing(preprocess.case_folding(query))))).strip() for query in query_asis]
    for query in query_asis:
        case_folding = preprocess.case_folding(query)
        cleansing = preprocess.cleansing(case_folding)
        normalization = preprocess.normalization(cleansing)
        filtering = preprocess.filtering(normalization)
        stemming = preprocess.stemming(filtering)
        tokenize = preprocess.tokenize(stemming)
    prediction = model.predict(query_processed_list)[0]

    if prediction == 1:
        klasifikasi = 'Positif'
    elif prediction == 0:
        klasifikasi = 'Negatif'

    return render_template("index.html", case_folding=case_folding, cleansing=cleansing, normalization=normalization, filtering=filtering, stemming=stemming, tokenize=tokenize,  prediction_text=klasifikasi)
