# Install relevant packages
import pandas as pd
import gc

# Load the data
# data_file_path = "C:\Users\Bas\OneDrive\MSc. Artificial Intelligence VU\MSc. AI Year 1\Data Mining Techniques\Assignment 2\dmt-2\data_subset"
train = pd.read_csv("data_subset/train_subset.csv", sep=',')
test = pd.read_csv("data_subset/test_subset.csv", sep=',')

def preprocess(data, kind):
    gc.collect()

if __name__ == "__main__":
    preprocess(data,kind)