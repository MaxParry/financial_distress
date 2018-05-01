import numpy as np
import pandas as pd

from sklearn.externals import joblib
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from flask import jsonify

from processing.modelTuning import shapeItUp

def makePredictions(valueDict):

    # load models
    rf_model = joblib.load('model/rf_model_v1.pkl')
    gbm_model = joblib.load('model/gbm_model_v1.pkl')
    X_scaler = joblib.load('model/fitted_X_scaler_v1.pkl')

    # Create
    df = pd.DataFrame(valueDict,index=[0])
    df = shapeItUp(df)

    X = df
    X_scaled = X_scaler.transform(X)

    rf_class_prediction = rf_model.predict(X_scaled)
    rf_probability_prediction = rf_model.predict_proba(X_scaled)

    gbm_class_prediction = gbm_model.predict(X_scaled)
    gbm_probability_prediction = gbm_model.predict_proba(X_scaled)

    def findAvg(value1,value2):
        return (value1+value2)/2

    predictions = {
        'rf_class': rf_class_prediction[0].item(),
        'gbm_class': gbm_class_prediction[0].item(),
        'probability': findAvg(rf_probability_prediction[0][0].item(),gbm_probability_prediction[0][0].item())
    }

    return predictions
