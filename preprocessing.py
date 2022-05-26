# Install relevant packages
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit

def preprocess(df):
    # something
    df = drop_irrelevant_features(df)
    df = remove_outliers(df)
    # Add engineered features
    df = add_datetime_features(df)
    df = add_price_order(df)
    # Normalize features
    for target in ['prop_starrating', 'prop_review_score', 'prop_location_score1', 'prop_location_score2', 'price_usd']:
        df[target + '_normalized'] = normalize(df, 'srch_id', target)
    return df

def load_train(train_path, val_split=False):
    df = pd.read_csv(train_path, parse_dates=['date_time'])
    df['score'] = calculate_score(df)

    df = preprocess(df)

    # remove columns missing in test data
    X, y = df.drop(['date_time', 'position', 'score', 'click_bool', 'booking_bool', 'gross_bookings_usd'], axis=1), df[['srch_id', 'score']]

    if val_split:
        groups = df['srch_id']
        return train_val_split(X, y, groups)
    return X, y

def load_test(test_path):
    df = pd.read_csv(test_path, parse_dates=['date_time'])
    df = preprocess(df)
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


def drop_irrelevant_features(df):
    df_temp = df
    for col in ['date_time', 'position', 'score', 'click_bool', 'booking_bool', 'gross_bookings_usd']:
        if col in df_temp.columns:
            df_temp = df_temp.drop(col, axis=1)
        else:
            continue
    feats = df_temp.columns[df_temp.isna().sum()/len(df_temp) * 100 > 90].to_list() # drop cols with more than 90% of data missing
    print("Dopping columns: ", feats)
    # feats = ['comp1_rate', 'comp1_inv', 'comp1_rate_percent_diff', 'comp3_rate_percent_diff', 'comp4_rate', 'comp4_inv', 'comp4_rate_percent_diff', 'comp6_rate', 'comp6_inv', 'comp6_rate_percent_diff', 'comp7_rate', 'comp7_inv', 'comp7_rate_percent_diff']
    return df.drop(feats, axis=1)

def remove_outliers(df):
    df = df[df['price_usd'] <= 600]
    df = df[df['srch_length_of_stay'] <= 10]
    df = df[df['srch_booking_window'] <= 250]
    df = df[df['srch_adults_count'] <= 6]
    df = df[df['srch_children_count'] <= 4]
    df = df[df['srch_room_count'] <= 4]
    return df

def normalize(df, group, target):
    groups = df.groupby(group)[target]
    # computes group-wise mean/std,
    # then auto broadcasts to size of group chunk
    mean = groups.transform("mean")
    std = groups.transform("std")
    target_normalized = (df[target] - mean) / std
    return target_normalized

def calculate_score(df):
    """
    calculates final score used to predict rank
    """
    df['booking_bool'] *= 5.0
    df['click_bool'] *= 1.0
    score = df[['booking_bool', 'click_bool']].max(axis=1)
    return score

def add_datetime_features(df):
    df['month'] = df['date_time'].dt.month
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['hour'] = df['date_time'].dt.hour
    return df

def add_price_order(df):
    df['price_order'] = df.groupby('srch_id')['price_usd'].apply(lambda x: np.argsort(x)[::-1]).values
    return df