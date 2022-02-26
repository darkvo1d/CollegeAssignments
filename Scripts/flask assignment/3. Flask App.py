# -*- coding: utf-8 -*-
"""
Created on Mon Dec 14 12:37:22 2020

@author: noopa
updated by: Sarvang
"""

import numpy as np
import pickle
import pandas as pd
from flask import Flask, request

app = Flask(__name__)
pickle_in = open("../assets/log_reg.pkl", "rb")
classifier = pickle.load(pickle_in)


@app.route('/')
def hello():
    return "Welcome All to Week-14"


@app.route('/predict')
def predict_class():
    radius_mean = request.args.get('radius_mean')
    texture_mean = request.args.get('texture_mean')
    perimeter_mean = request.args.get('perimeter_mean')
    area_mean = request.args.get('area_mean')
    smoothness_mean = request.args.get('smoothness_mean')
    compactness_mean = request.args.get('compactness_mean')
    concavity_mean = request.args.get('concavity_mean')
    concave_points_mean = request.args.get('concave points_mean')
    symmetry_mean = request.args.get('symmetry_mean')
    fractal_dimension_mean = request.args.get('fractal_dimension_mean')

    radius_se = request.args.get('radius_se')
    texture_se = request.args.get('texture_se')
    perimeter_se = request.args.get('perimeter_se')
    area_se = request.args.get('area_se')
    smoothness_se = request.args.get('smoothness_se')
    compactness_se = request.args.get('compactness_se')
    concavity_se = request.args.get('concavity_se')
    concave_points_se = request.args.get('concave points_se')
    symmetry_se = request.args.get('symmetry_se')
    fractal_dimension_se = request.args.get('fractal_dimension_se')

    radius_worst = request.args.get('radius_worst')
    texture_worst = request.args.get('texture_worst')
    perimeter_worst = request.args.get('perimeter_worst')
    area_worst = request.args.get('area_worst')
    smoothness_worst = request.args.get('smoothness_worst')
    compactness_worst = request.args.get('compactness_worst')
    concavity_worst = request.args.get('concavity_worst')
    concave_points_worst = request.args.get('concave points_worst')
    symmetry_worst = request.args.get('symmetry_worst')
    fractal_dimension_worst = request.args.get('fractal_dimension_worst')

    prediction = classifier.predict([[radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean,
                                      fractal_dimension_mean, radius_se, texture_se, perimeter_se, area_se, smoothness_se, compactness_se, concavity_se, concave_points_se,
                                      symmetry_se, fractal_dimension_se, radius_worst, texture_worst, perimeter_worst, area_worst, smoothness_worst, compactness_worst,
                                      concavity_worst, concave_points_worst, symmetry_worst, fractal_dimension_worst]])

    return " The Predicated Class is" + str(prediction)


@app.route('/predict_test', methods=["POST"])
def predict_test_class():
    df_test = pd.read_csv(request.files.get("file"))
    prediction = classifier.predict(df_test)
    return " The Predicated Class for the TestFile is" + str(list(prediction))


if __name__ == '__main__':
    app.run()
