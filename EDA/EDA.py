import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import datetime as dt
import sklearn

data_file_path = "C:/Users/Bas/OneDrive/MSc. Artificial Intelligence VU/MSc. AI Year 1/Data Mining Techniques/Assignment 2/data/"
training = pd.read_csv(data_file_path + "training_set_VU_DM.csv",sep=',')
len(training)
