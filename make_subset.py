"""
This files makes subsets from the original train and test data file for computational optimization
"""

import pandas as pd

def make_subset():
    orginal_path = r"C:\Users\Bas\OneDrive\MSc. Artificial Intelligence VU\MSc. AI Year 1\Data Mining Techniques\Assignment 2\data"

    # make training subset
    print("Reading training data ...")
    train = train = pd.read_csv(orginal_path + "/training_set_VU_DM.csv", sep=',')
    print("Making train_subset...")
    train_subset = train[train['srch_id'] < train['srch_id'].unique()[10000]]
    train_subset.to_csv("data_subset/train_subset.csv", index=False, header=True)
    del train, train_subset

    # make test subset
    print("Reading test data ...")
    test = pd.read_csv(orginal_path + "/test_set_VU_DM.csv", sep=',')
    print("Making test_subset...")
    test_subset = test[test['srch_id'] < test['srch_id'].unique()[10000]]
    test_subset.to_csv("data_subset/test_subset.csv", index=False, header=True)
    del test, test_subset

    print("Finished!")

if __name__ == '__main__':
    make_subset()
