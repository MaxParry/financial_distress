#################################################
# Dependencies
#################################################
from flask import Flask, render_template, jsonify, redirect
from flask import Response

import sqlalchemy
from sqlalchemy.ext.automap import automap_base
from sqlalchemy.ext.declarative import declarative_base #classes into tables
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy import create_engine, func, inspect, Column, Integer, String

import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.externals import joblib
from sklearn.datasets import make_regression

from processing.result import makePredictions
# from processing.modelTuning import shapeItUp

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# Full dashboard
@app.route('/')
def index():
    """Return the  homepage."""

    return render_template('index.html')

@app.route('/about')
def about():
    """Return the about page."""

    return render_template('about.html')


@app.route('/api/v1.0/<age>/<salary>/<dependents>/<creditLine>/<creditLimit>/<realEstate>/<monthlySpend>/<totalDebt>/<overdue>')
def getResult(age,salary,dependents,creditLine,creditLimit,realEstate,monthlySpend,totalDebt,overdue):

    valueDict = {
                    'age': int(age),
                    'MonthlyIncome': (int(salary)/12),
                    'NumberOfDependents': int(dependents),
                    'NumberOfOpenCreditLinesAndLoans': int(creditLine),
                    'RevolvingUtilizationOfUnsecuredLines': (int(totalDebt)/int(creditLimit)),
                    'NumberRealEstateLoansOrLines': int(realEstate),
                    'DebtRatio': (int(monthlySpend)/(int(salary)/12)),
                    'NumTimesPastDue': int(overdue),
                    'MonthlyCosts' : int(monthlySpend)
                }

    print(valueDict)

    result = makePredictions(valueDict)

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
