# Install relevant packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import warnings
import datetime as dt

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)

from pandas.core.common import SettingWithCopyWarning
warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)
warnings.simplefilter(action="ignore", category=FutureWarning)


# Load the data
data_file_path = "C:/Users/Bas/OneDrive/MSc. Artificial Intelligence VU/MSc. AI Year 1/Data Mining Techniques/Assignment 2/data/"
train = pd.read_csv(data_file_path + "training_set_VU_DM.csv", sep=',')