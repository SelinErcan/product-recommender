import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
import datetime
import matplotlib.pyplot as plt
import seaborn as sns

def create_cos_sim_matrix():

    # get prapared sales data
    sales_df = pd.read_pickle("./data/sales.pkl")

    # convert df into a pivot matrix where each row is an session and each column is a productid and the values are the counts of the products in each of the session
    pivot_df = pd.pivot_table(sales_df, index = 'sessionid',columns = 'productid', values = 'name', aggfunc = 'count')

    # fill null values in the matrix with zeros
    pivot_df.reset_index(inplace=True)
    pivot_df = pivot_df.fillna(0)
    pivot_df = pivot_df.drop('sessionid', axis=1)

    # transform our pivot table into a co-occurrence matrix
    co_matrix = pivot_df.T.dot(pivot_df)
    np.fill_diagonal(co_matrix.values, 0)

    # transform the co-occurrence matrix into a matrix of cosine similarities
    cos_score_df = pd.DataFrame(cosine_similarity(co_matrix))
    cos_score_df.index = co_matrix.index
    cos_score_df.columns = np.array(co_matrix.index)

    print(cos_score_df.head(5))

    cos_score_df.to_pickle("./models/cosine_similarity.pkl")

if __name__ == "__main__":
    create_cos_sim_matrix()