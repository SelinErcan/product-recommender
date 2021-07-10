from data_helper import prepare_data
from cosine_similarity import create_cos_sim_matrix
from predict import recommend_product
import numpy as np
import pandas as pd
from sklearn import preprocessing
import json
import os

MODEL_DIR = 'models'

# Create model directory if doens't exist
try:
    os.mkdir(MODEL_DIR)
    print("Directory " , MODEL_DIR ,  " Created ") 
except FileExistsError:
    print("Directory " , MODEL_DIR ,  " already exists")

prepare_data()
create_cos_sim_matrix()
