import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flask import jsonify

from modelTuning import shapeItUp,add_incomeperdependent_column,add_indicator_column

def makePredictions(valueDict):

    model = joblib.load('trained_default_model_v1.pkl')

    df = pd.DataFrame(valueDict,index=[0])
    df = shapeItUp(df)

    X = df
    X_scaler = StandardScaler().fit(X)
    X_train_scaled = X_scaler.transform(X)

    class_prediction = model.predict(X_train_scaled)
    probability_prediction = model.predict_proba(X_train_scaled)

    predictions = {
        'class': class_prediction[0].item(),
        'probability': probability_prediction[0][1].item()
    }

    return jsonify(predictions)
