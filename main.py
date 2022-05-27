import numpy as np
import pandas as pd
import argparse

from evaluation import train_model
from preprocessing import load_train, load_test
from submit import make_submission


def run(train_path, test_path, model_name='pairwise', param_space={}):
    print(train_path, test_path)
    print("Loading data...")
    X_train, y_train = load_train(train_path)
    X_test = load_test(test_path)

    print("Training model...")
    model = train_model(X_train, y_train, model_name=model_name, param_space=param_space)
    
    print("Making Submission...")
    make_submission(X_test, model)
    
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='pairwise', type=str,
                        help='What model to use for the hyperparameter tuning',
                        choices=['pairwise', 'listwise_ndcg', 'listwise_map'])
    parser.add_argument('--train_path', type=str, default='data/training_set_VU_DM.csv',
                    help='Specifies location of the training data file')
    parser.add_argument('--test_path', type=str, default='data/test_set_VU_DM.csv',
                    help='Specifies location of the testing data file')

    args = parser.parse_args()
    #TODO set the param_space to the best hyperparameters resulting from hyperparameter search
    param_space = {}
    model = run(args.train_path, args.test_path, args.model_name, param_space)
