# Install relevant packages
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from config import TRAINING_PATH, TEST_PATH

def transform(df):
    return df

def calculate_score(df):
    """
    calculates final score used to predict location
    """
    score = df['booking_bool'] * 5 + df['click']
    return score

def train_val_split(X, y, val_size):
    """
    could be more thoughtful, maybe split after grouping srch ids?
    """
    return train_test_split(X, y, test_size=val_size)

# Load the data
def load_train_val():
    train = pd.read_csv(TRAINING_PATH)

    # Remove travel agents
    # Get all unique user ids
    unique_users = train.user_id.unique()
    # Remove all non-bookings to make counting easier
    t1 = train[train.is_booking != 0]
    for user in unique_users:
        # Count the number of rows under a single user
        bookings = len(t1.loc[t1['user_id'] == user])
        if bookings >= 20:
            # Remove the travel agent from dataset
            train = train[train.user_id != user]

    train['score'] = calculate_score(train)

    X = train.drop(['srch_id', 'position', 'score', 'click_bool', 'booking_bool', 'gross_booking_usd'])
    y = train['score']

    X_train, 
    return X_train, y_train, X_val, y_val