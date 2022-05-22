# Install relevant packages
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

from features import *
from config import TRAINING_PATH, TEST_PATH


def add_features(df):
    # df = remove_travel_agents(df)
    # df = add_datetime_features(df)
    return df

def normalize(df):
    return df

def calculate_score(df):
    """
    calculates final score used to predict rank
    """
    df['booking_bool'] *= 5.0
    score = df[['booking_bool', 'click_bool']].max(axis=1)
    return score

def train_val_split(X, y, groups, val_size=.7):
    """
    Splits training data based on groups
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=42)
    train_indices, val_indices = next(gss.split(X, y, groups))

    X_train, y_train = X.loc[train_indices], y.loc[train_indices]
    X_val, y_val = X.loc[val_indices], y.loc[val_indices]

    print("Training / Validation shape:")
    print((X_train.shape, y_train.shape), (X_val.shape, y_val.shape))
    return X_train, y_train, X_val, y_val

def load_train_val():
    df = pd.read_csv(TRAINING_PATH, parse_dates=['date_time'])
    df['score'] = calculate_score(df)

    # Add engineered features
    df = add_features(df)
    # Normalize features
    df = normalize(df)

    X, y = df.drop(['date_time', 'position', 'score', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1), df[['srch_id', 'score']]
    groups = df['srch_id']
    return train_val_split(X, y, groups)

def load_test():
    df = pd.read_csv(TEST_PATH, parse_dates=['date_time'])
    df = add_features(df)
    X = df.drop(['date_time'], axis=1)
    return X

if __name__ == "__main__":
    x1,y1, x2, y2 = load_train_val()
    print(x1)