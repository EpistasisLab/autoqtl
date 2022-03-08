import sys
sys.path.append("C:/Users/ghosha/.vscode/autoqtl")
import autoqtl

from autoqtl.builtins.feature_encoding_frequency_selector import FeatureEncodingFrequencySelector

import numpy as np
import pandas as pd

test_data = pd.read_csv("tests/randomset3.csv")
print(test_data.shape)

test_data_numpyarray = pd.DataFrame(test_data).to_numpy()

test_X = test_data_numpyarray[:, : -1 ]

op = FeatureEncodingFrequencySelector(0.05)
op.fit(test_X, y=None)

transformed_X = op.transform(test_X)

print( transformed_X.shape[0] == test_X.shape[0])
print( transformed_X.shape[1] != test_X.shape[1])
print(transformed_X.shape[1])
print(op.selected_feature_indexes)