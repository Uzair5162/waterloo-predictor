from flask import Flask, render_template, jsonify, request
from sklearn.externals import joblib
import numpy as np

app = Flask(__name__)
model = None

def load_model():
    global model
    if not model:
        model = joblib.load("waterloo_classifier.pkl")
    return model

@app.route('/')
def index():
    values = request.values

    return render_template('index.html', form_values=values)

@app.route('/process_form', methods=["POST"])
def process_form():
    values = {
        'avg': int(request.form['avg']),
        'aif': int(request.form['aif']),
        'int': int(request.form['int']),
        'adj': int(request.form['adj']),
        'sex': int(request.form['sex'])
    }

    model = load_model()
    model_params = [[
        values['avg'],
        values['aif'],
        values['int'],
        values['adj'],
        values['sex']
    ]]

    prediction = model.predict(model_params)[0]
    probabilities = model.predict_proba(model_params)[0]

    return render_template('results.html', prediction=prediction, probabilities=probabilities, form_values=values)

if __name__ == "__main__":
    app.run(debug=True)