# Install relevant packages
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def load_train(train_path, val_split=False):
    df = pd.read_csv(train_path, parse_dates=['date_time'])
    df['score'] = calculate_score(df)

    # Add engineered features
    df = add_features(df)
    # Normalize features
    df = normalize(df)

    X, y = df.drop(['date_time', 'position', 'score', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1), df[['srch_id', 'score']]
    
    if val_split:
        groups = df['srch_id']
        return train_val_split(X, y, groups)
    return X,y

def load_test():
    df = pd.read_csv("data/test_set_VU_DM.csv", parse_dates=['date_time'])
    df = add_features(df)
    X = df.drop(['date_time'], axis=1)
    return X

def train_val_split(X, y, groups, val_size=.7):
    """
    Splits training data based on groups
    """
    gss = GroupShuffleSplit(n_splits=1, train_size=val_size, random_state=42)
    train_indices, val_indices = next(gss.split(X, y, groups))

    X_train, y_train = X.loc[train_indices], y.loc[train_indices]
    X_val, y_val = X.loc[val_indices], y.loc[val_indices]

    return X_train, y_train, X_val, y_val

def add_features(df):
    df = add_datetime_features(df)
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

def add_datetime_features(df):
    df['month'] = df['date_time'].dt.month
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['hour'] = df['date_time'].dt.hour
    return df
