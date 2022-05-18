# Install relevant packages
import numpy as np
import pandas as pd

from sklearn.model_selection import GroupShuffleSplit
from config import TRAINING_PATH, TEST_PATH

def enrich(df):
    # Remove travel agents
    # Get all unique user ids
    unique_users = df.user_id.unique()
    # Remove all non-bookings to make counting easier
    t1 = df[df.is_booking != 0]
    for user in unique_users:
        # Count the number of rows under a single user
        bookings = len(t1.loc[t1['user_id'] == user])
        if bookings >= 20:
            # Remove the travel agent from dataset
            df = df[df.user_id != user]
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

# Load the data
def load_train_val():
    df = pd.read_csv(TRAINING_PATH)
    df['score'] = calculate_score(df)

    groups = df['srch_id']
    X = df.drop(['date_time', 'position', 'score', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1)
    y = df[['srch_id', 'score']]
    
    return train_val_split(X, y, groups)

def load_test():
    df = pd.read_csv(TEST_PATH)
    X = df.drop(['srch_id', 'position', 'score', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1)
    return X

if __name__ == "__main__":
    load_train_val()