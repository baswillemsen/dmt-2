# Install relevant packages
import numpy as np
import pandas as pd

from config import TRAINING_PATH, TEST_PATH

def transform(df):
    return df

# Load the data
def train_test_split():
    train = pd.read_csv(TRAINING_PATH)
    test = pd.read_csv(TEST_PATH)
    return train, test