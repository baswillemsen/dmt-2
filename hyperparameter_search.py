import numpy as np
import pandas as pd
from hyperopt import hp, fmin,tpe, Trials
from functools import partial
from evaluation import run
import argparse

def evaluate_model(param_space, train_path, model_name, gpu):
    #evaluates the model with the given hyperparameters. Return 1-ndgc@k as fmin from the hyperopt library minimizes that score that is returned.
    param_space['colsample_bytree'] = param_space['colsample_by']
    param_space['colsample_bylevel'] = param_space['colsample_by']
    param_space['colsample_bynode'] = param_space['colsample_by']
    del param_space['colsample_by']
    if gpu is not None:
        param_space['gpu_id'] = gpu
        param_space['tree_method'] = 'gpu_hist'
    _, validation_score = run(train_path, model_name, param_space)
    return 1.0-validation_score

def hypertune(model_name, train_path, param_dict, gpu=None):
    seed = 42
    rstate = np.random.default_rng(seed)
    param_space = {}
    for param_name, param_list in param_dict.items():
        param_space[param_name] = hp.choice(param_name, param_list)
    
    optimization_function = partial(
                                evaluate_model,
                                train_path = train_path,
                                model_name = model_name,
                                gpu = gpu)
    trials = Trials()
    hopt = fmin(fn=optimization_function,
                space=param_space,
                algo=tpe.suggest,
                max_evals=15,
                rstate=rstate,
                trials=trials)
    print('Best parameters:', hopt)
    print('All tried out parameters:')
    for i, x in enumerate(trials.trials):
        print('trial', i)
        values = x['misc']['vals']
        print_string = ''
        for param_key, param_list in param_dict.items():
            print_string += f'{param_key}: {param_list[values[param_key][0]]}. '
        print(print_string)
        print(x['result'])

if __name__ == '__main__':
    # Feel free to add more argument parameters
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    parser.add_argument('--model_name', default='pairwise', type=str,
                        help='What model to use for the hyperparameter tuning',
                        choices=['pairwise', 'listwise_ndcg', 'listwise_map'])
    parser.add_argument('--train_path', default='data/training_set_VU_DM.csv', type=str,
                        help='Specifies location of the training data file')
    parser.add_argument('--gpu_node', default=-1, type=int,
                        help='Specifies GPU node')

    args = parser.parse_args()
    if args.gpu_node == -1:
        args.gpu_node = None

    param_dict = {'eta': [0.01, 0.05, 0.1, 0.15, 0.2],
                    'gamma' : [0.0, 0.05, 0.1, 0.5, 1.0, 2.0],
                    'max_depth' : [3, 5, 6, 7, 8, 10],
                    'min_child_weight' : [0.0, 0.05, 0.1, 0.5, 1.0, 2.0],
                    'max_delta_step' : [0, 1, 3, 5, 7],
                    'subsample' : [0.5, 0.7, 0.9, 1.0],
                    'colsample_by' : [0.5, 0.7, 0.9, 1.0],
                    'lambda' : [0.2, 0.4, 0.6, 0.8, 1.0],
                    'alpha' : [0.2, 0.4, 0.6, 0.8, 1.0]}
    hypertune(args.model_name, args.train_path, param_dict, args.gpu_node)