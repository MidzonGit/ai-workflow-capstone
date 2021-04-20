# -*- coding: utf-8 -*-
"""
Created on Sat Apr 17 11:14:04 2021

@author: MithunMohan
"""

from flask import Flask, jsonify, request, Response
from flask import render_template, send_from_directory
import os
import re
import numpy as np
import pandas as pd

from utils.model import model_train, model_predict

app = Flask(__name__)


@app.route("/")
def home():
    return '''Welcome to the AI workflow Capstone server. 
              Use /predict endpoint and pass Country, Date information to receive predictions 
              Use /train endpoint and pass Environment value to kickstart model training 
              Use /logs endpoint alongwith url param type = train or predict for logfile to retrieve log files'''


@app.route('/train', methods=['GET'])
def train_get():
    return '''<form method="POST">
                   <div><label>Env - Train or Production: <input type="text" name="dir"></label></div>
                   <input type="submit" value="Submit">
               </form>'''


@app.route('/train', methods=['POST'])
def train_post():
    if not request.json:
        msg="Did not receive request data"
        return jsonify(msg)
    if 'dir' not in request.json:
        msg="Directory argument not found in the request"
        return jsonify(msg)
    txt = request.form.get['dir']
    if txt.lower() not in ['train','production']:
        msg="Directory can only be Train or Production"
        return jsonify(msg)
    dir_name=r'data\cs-{}'.format(txt.lower())
    model_train(dir_name)
    print("Model training completed!")#log rather
    return (jsonify("Model training completed for environment {}".format(txt)))


@app.route('/predict', methods=['GET'])
def predict_get():
    return '''<form method="POST">
                   <div><label>Enter Country: <input type="text" name="country"></label></div>
                   <div><label>Enter Date in YYYY-MM-DD format: <input type="text" name="date"></label></div>
                   <input type="submit" value="Submit">
               </form>'''


@app.route('/predict', methods=['POST'])
def predict_post():
    if not request.json:
        msg="Did not receive request data"
        return jsonify(msg)
    if 'country' not in request.json:
        msg="No Country found in request"
        return jsonify(msg)
    if 'date' not in request.json:
        msg="No Date found in request"
        return jsonify(msg)
    date = request.form.get('date')
    if 'date' in request.json and int(date.split('-')[0]) not in [2018,2019]:
        msg="Valid date not entered. Choose an year between 2018 and 2019"
        return jsonify(msg)
    if 'date' in request.json and 1<=int(date.split('-')[1])<=12:
        msg="Valid date not entered. Month value invalid"
        return jsonify(msg)
    if 'date' in request.json and 1<=int(date.split('-')[2])<=31:
        msg="Valid date not entered. Day value invalid"
        return jsonify(msg)
    #set test flag
    year = date.split('-')[0]
    month = date.split('-')[1]
    day = date.split('-')[2]
    country = request.form.get('country')
    prediction = model_predict(country, year, month, day, test=True)
    output = {}
    for key,item in prediction.items():
        if isinstance(item,np.ndarray):
            output[key] = item.tolist()
        else:
            output[key] = item
    output_text = "{0}: Predicted Forecast for 30 day period on {1} is: {2}".format(country,date,output)
    return jsonify(output_text)

        
@app.route("/logs/<type>")
def retrieveLogs(typ):
    if typ not in ['train','predict']:
        return jsonify('Wrong parameter. Values need to be either train or predict')
    path = r'logs/{}.log'.format(typ)
    if not os.path.isfile(path):
        print("ERROR: API (log): cannot find log file")
        return jsonify('Log file does not exist')
    filename='{}.log'.format(typ)
    try:
        file = open(path,'r')
        returnfile = file.read().encode('latin-1')
        file.close()
        return Response(returnfile,mimetype="text/plain",headers={"Content-disposition":"attachment; filename=model.log"})
    except Exception as e:
        print(str(e))
        return jsonify([])



if __name__ == '__main__':
    debug=True
    app.run(host='127.0.0.1', debug=debug, port=8080)
    