from data_helper import prepare_data
from cosine_similarity import create_cos_sim_matrix
import numpy as np
import pandas as pd
from sklearn import preprocessing
import json
import os
from flask.json import jsonify

def evaluate(categories, predictions):
    """Evaluate model by finding hit ratio of categories"""

    total = len(predictions)
    hit = 0

    for pred in predictions:
        if pred in categories:
            hit += 1
    print("Model Score (rate of same category prediction) : %", hit/total*100)


def recommend_product(items):
    """
    Predict related products by cosine similarity matrix
    input: item's productids in cart as list
    output: dict of related items as json 
    """

    sales_df = pd.read_pickle("./data/sales.pkl")

    items_in_cart = sales_df[sales_df.productid.isin(items)][['productid', 'brand', 'category', 'subcategory', 'price']].drop_duplicates()
    print("\nItems in cart:\n",items_in_cart)

    cos_score_df = pd.read_pickle("./models/cosine_similarity.pkl")

    predicted_values = cos_score_df[cos_score_df.index.isin(items)]
    recommended_items = []
    for value in predicted_values.index.values:
        recommended_items.extend(predicted_values[predicted_values.index == value].T.sort_values(ascending=False, by=value)[1:11].index.values)
    recommended_items = list(set(recommended_items))

    info_df = sales_df[['productid', 'brand', 'price', 'category', 'subcategory']].drop_duplicates()
    recommended_items = info_df[info_df.productid.isin(recommended_items)].sample(10)
    print("\nRecommended_items:\n",recommended_items)
    evaluate(items_in_cart.category.values, recommended_items.category.values)

    recommended_items = recommended_items.reset_index(drop=True).to_dict('records')
    
    return jsonify({'recommended_items': recommended_items})

if __name__ == "__main__":
    recommend_product(items = ["ZYECZACI9300276", "HBV00000AX6LR"])
