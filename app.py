# -*- coding: utf-8 -*-
"""
Created on Tue Sep 22 13:06:32 2020

@author: AKSHAY
"""

from flask import Flask, request
import pandas as pd
import numpy as np
import pickle
import flasgger
from flasgger import Swagger

app=Flask(__name__)
Swagger(app)

pickle_in=open('classifier.pkl','rb')
classifier=pickle.load(pickle_in)


@app.route('/')
def welocome_page():
    return "Welcome everybody"

@app.route('/predict')
def predict_note_authentication():
    
    """Let's authenticate the bank note.
    This is bank note Using docstrings for specifications.
    
    ---
    parameters:
        - name: variance
          in: query
          type: number
          required: true
        - name: skewness
          in: query
          type: number
          required: true
        - name: curtosis
          in: query
          type: number
          required: true
        - name: entropy
          in: query
          type: number
          required: true
    responses:
        200:
            description: The output values
    
    """
    variance=request.args.get("variance")
    skewness=request.args.get("skewness")
    curtosis=request.args.get("curtosis")
    entropy=request.args.get("entropy")
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    return "The predicted value is "+ str(prediction)

@app.route('/predict_file', methods=["POST"])
def predict_note_file():
    """Let's authenticate the bank note.
    This is bank note Using docstrings for specifications.
    ---
    parameters:
        - name: file
          in: formData
          type: file
          required: true
    responses:
        200:
            description: The output values
    
    
    
    """
    df_test=pd.read_csv(request.files.get("file"))
    prediction=classifier.predict(df_test)
    return "The predicted value for CSV is: "+ str(list(prediction))


if __name__=='__main__':
    app.run()