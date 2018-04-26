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

#################################################
# Flask Setup
#################################################
app = Flask(__name__)

# Full dashboard
@app.route('/')
def index():
    """Return the dashboard homepage."""

    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True)
