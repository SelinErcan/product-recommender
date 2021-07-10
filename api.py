import flask
from flask import Flask, make_response, render_template, request,send_from_directory, send_file
from flask_cors import CORS
from requests.exceptions import (ConnectionError, HTTPError, RequestException, Timeout)
import os
import shutil
import json
import requests
import time
from predict import recommend_product

app = Flask(__name__)

@app.route('/recommend',methods=['GET', 'POST'])
def recommend():

    try:
        # check whether json file exists
        if request.get_json() is None:
            return api_error('Request does not contain a json file!', 400)

        # request json file
        json_response = request.get_json()

        json_file = recommend_product(json_response['productids_in_cart'])

        return json_file
    
    except (ConnectionError, HTTPError) as e:
        return api_error(str(e), 500)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int("5000"), debug=True)
