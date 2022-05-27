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
    parser.add_argument('--gpu_node', default=-1, type=int,
                        help='Specifies GPU node')

    args = parser.parse_args()
    #TODO set the param_space to the best hyperparameters resulting from hyperparameter search
    param_space = {'eta': 0.2,
                    'gamma' : 0.0,
                    'max_depth' : 5,
                    'min_child_weight' : 1.0,
                    'max_delta_step' : 5,
                    'subsample' : 1.0,
                    'colsample_bytree' : 0.7,
                    'colsample_bylevel' : 0.7,
                    'colsample_bynode' : 0.7,
                    'lambda' : 1.0,
                    'alpha' : 0.6}
    if args.gpu_node != -1:
        param_space['gpu_id'] = args.gpu_node
        param_space['tree_method'] = 'gpu_hist'
    model = run(args.train_path, args.test_path, args.model_name, param_space)
