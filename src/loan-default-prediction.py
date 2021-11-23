# packages
from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import clear_output
import urllib
import tensorflow as tf
from tensorflow import feature_column as fc

# loading the data
train = pd.read_csv('../data/training.csv')  # 75% data used to train model
evaluate = pd.read_csv('../data/testing.csv')  # 25% data used to test model
y_train = train.pop('Defaulted?')  # variable to be predicted will be taken
y_evaluate = evaluate.pop('Defaulted?')

# classify the data using featured columns
CATEGORICAL_COLUMNS = ['Employed']
NUMERIC_COLUMNS = ['Bank Balance', 'Annual Salary']

feature_columns = []
for feature_name in CATEGORICAL_COLUMNS:
    categories = train[feature_name].unique()  # returns list of all unique values
    feature_columns.append(tf.feature_column.categorical_column_with_vocabulary_list(feature_name, categories))

for feature_name in NUMERIC_COLUMNS:
    feature_columns.append(tf.feature_column.numeric_column(feature_name, dtype=tf.float32))
