import argparse

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import BayesianRidge
from utils import get_df_per_location, create_lagged_feature, cut_top_rows


#For strategies that train separate models per region, this function can be used as is
def train(csv_fn, model_fn):
    models = {}
    locations = get_df_per_location(csv_fn)
    for location,df in locations.items():
        model = train_single_region(df, location)
        models[location] = model

    joblib.dump(models, model_fn)


def train_single_region(df, location):
    #features = ['rainfall', 'mean_temperature', 'last_rainfall', 'last_disease_cases']
    features = ['rainfall', 'mean_temperature']
    X = df[features].copy()
    df['disease_cases'] = df['disease_cases'].fillna(0)  # set NaNs to zero (not a good solution, just for the example to work)
    Y = df['disease_cases']
    # X['last_rainfall'] = df['rainfall'].shift(1)
    # X['last_disease_cases'] = df['disease_cases'].shift(1)
    create_lagged_feature(X, 'mean_temperature', 1)
    create_lagged_feature(X, 'rainfall', 1)
    create_lagged_feature(X, 'disease_cases', 1, df)
    X = cut_top_rows(X, 1)
    Y = cut_top_rows(Y, 1)

    model = BayesianRidge()
    # Convert to numpy with explicit dtype for consistency with prediction
    X_train = X.to_numpy(dtype=np.float64, copy=True)
    Y_train = Y.to_numpy(dtype=np.float64, copy=True)
    model.fit(X_train, Y_train)
    # model.last_disease_cases = Y.iloc[-1]
    # model.last_rainfall = X['rainfall'].iloc[-1]
    return model

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train a minimalist forecasting model.')

    parser.add_argument('csv_fn', type=str, help='Path to the CSV file containing input data.')
    parser.add_argument('model_fn', type=str, help='Path to save the trained model.')
    args = parser.parse_args()
    train(args.csv_fn, args.model_fn)






