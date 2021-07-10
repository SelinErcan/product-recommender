import numpy as np
import pandas as pd
from sklearn import preprocessing
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns

def plot_countplot(data, x, hue, title):
    """Plot countplot for categories"""
    plt.figure(figsize=(50, 20))
    plt.rcParams['font.size'] = '20'
    palette = sns.color_palette("deep")
    sns.countplot(x=x, hue=hue, data=data, palette=palette)
    plt.legend(fontsize=20, loc='upper right', title='Categories')
    plt.title(title)
    plt.savefig('./data/' + title + '.png')
    plt.close()

def read_json(file_name, name):
    """Read json files as dataframe"""

    with open(file_name) as f:
        d = json.load(f)

    df = pd.json_normalize(d[name])
    
    print("Data shape:{} of {}".format(df.shape, file_name))
    
    return df

def prepare_data():
    """Prepare data for the model"""

    # read json files
    events_df = read_json("data/events.json", "events")
    meta_df = read_json("data/meta.json", "meta")

    # merge two df
    sales_df = events_df.merge(meta_df, how='left', on='productid')

    # convert eventtime column to datetime
    sales_df['eventtime'] = pd.to_datetime(sales_df['eventtime'])

    # split time
    sales_df["hour"] = sales_df['eventtime'].map(lambda x: x.hour)
    sales_df["day"] = sales_df['eventtime'].map(lambda x: x.day)
    sales_df["month"] = sales_df['eventtime'].map(lambda x: x.month)
    sales_df["year"] = sales_df['eventtime'].map(lambda x: x.year)

    # drop features that contains null and constant value
    sales_df = sales_df.dropna()
    sales_df = sales_df.loc[:, (sales_df != sales_df.iloc[0]).any()] 

    # plot sales for each category per month
    plot_countplot(data=sales_df, x="month", hue="category", title='categories_per_month')

    # checkout categories sales size for each month
    categories = sales_df.category.unique()
    monthly_categories = sales_df.groupby(['month', 'category']).size()
    monthly_categories = monthly_categories.reset_index(name='size')

    # eleminate undersampled months
    sales_df = sales_df[sales_df.month==6]
    del sales_df['month']

    # plot sales for each category per day
    plot_countplot(data=sales_df, x="day", hue="category", title='categories_per_day')

    # eleminate undersampled days
    sales_df = sales_df[sales_df.day.isin(range(0,16))]

    sales_df.to_pickle("./data/sales.pkl")

if __name__ == "__main__":
    prepare_data()