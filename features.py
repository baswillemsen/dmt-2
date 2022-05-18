def add_datetime_features(df):
    df['month'] = df['date_time'].dt.month
    df['week'] = df['date_time'].dt.isocalendar().week
    df['year'] = df['date_time'].dt.year
    return df

def remove_travel_agents(df):
    """
    Remove travel agents defined as having more than 20 bookings
    """
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
