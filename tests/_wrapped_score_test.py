
import sys

import pandas as pd
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
sys.path.append("C:/Users/ghosha/.vscode/autoqtl")
import autoqtl

from autoqtl.gp_deap import _wrapped_score
from autoqtl.base import AUTOQTLBase
from autoqtl.autoqtl import AUTOQTLRegressor

# data
test_data = pd.read_csv("tests/randomset3.csv")
test_data_numpyarray = pd.DataFrame(test_data).to_numpy()

test_X = test_data_numpyarray[:, : -1 ]
test_y = test_data_numpyarray[:,-1]

features_dataset1, features_dataset2, target_dataset1, target_dataset2 = train_test_split(test_X, test_y, train_size=0.5, random_state=42)

autoqtl_obj = AUTOQTLRegressor(
    random_state=42,
    population_size=10,
    offspring_size=2,
    generations=1,
    verbosity=0
)

"""population = autoqtl_obj._toolbox.population(n=10)
autoqtl_obj._evaluate_individuals(population, features_dataset1, target_dataset1, features_dataset2, target_dataset2)"""
estimators = [('feature_extraction', VarianceThreshold(threshold=0.05)), ('regression', LinearRegression())]
pipeline = Pipeline(estimators)

score_returned = _wrapped_score(pipeline, features_dataset1, target_dataset1, autoqtl_obj.scoring_function)
print(score_returned)