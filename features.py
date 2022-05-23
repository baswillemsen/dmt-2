def add_datetime_features(df):
    df['month'] = df['date_time'].dt.month
    df['dayofweek'] = df['date_time'].dt.dayofweek
    df['hour'] = df['date_time'].dt.hour
    return df
