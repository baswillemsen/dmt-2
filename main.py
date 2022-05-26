import numpy as np
import pandas as pd
import argparse

from evaluation import train_model
from preprocessing import load_train, load_test
from submit import make_submission


def run(train_path, test_path):
    print("Loading data...")
    X_train, y_train = load_train(train_path)
    X_test = load_test(test_path)

    print("Training model...")
    model = train_model(X_train, y_train)

    print("Making Submission...")
    make_submission(X_test, model)
    
if __name__ == "__main__":
    subset = True

    if subset == True:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_path', type=str, default='data/train_subset.csv',
                        help='Specifies location of the training data file')
        parser.add_argument('--test_path', type=str, default='data/test_subset.csv',
                        help='Specifies location of the training data file')
    else:
        parser = argparse.ArgumentParser()
        parser.add_argument('--train_path', type=str, default='data/training_set_VU_DM.csv',
                        help='Specifies location of the training data file')
        parser.add_argument('--test_path', type=str, default='data/test_set_VU_DM.csv',
                        help='Specifies location of the training data file')

    args = parser.parse_args()
    model = run(args.train_path, args.test_path)