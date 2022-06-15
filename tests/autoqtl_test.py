import os, sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

from shutil import rmtree
import sys
from tempfile import mkdtemp

from sklearn.linear_model import LinearRegression

from sklearn.model_selection import train_test_split
sys.path.append("/home/ghosha/common/autoqtl/autoqtl")
import autoqtl
import autoqtl.base
from autoqtl.autoqtl import AUTOQTLRegressor
from autoqtl.base import AUTOQTLBase

from autoqtl.gp_types import Output_Array
from autoqtl.gp_deap import mutNodeReplacement, _wrapped_score, pick_two_individuals_eligible_for_crossover, cxOnePoint, varOr, initialize_stats_dict
from autoqtl.operator_utils import AUTOQTLOperatorClassFactory, set_sample_weight, source_decode

import numpy as np
import pandas as pd

from deap import creator, gp

autoqtl_obj = AUTOQTLRegressor()
autoqtl_obj._fit_init()

# dataset1
test_data = pd.read_csv("/home/ghosha/common/autoqtl/tests/BMIwTail.csv")

test_data_numpyarray = pd.DataFrame(test_data).to_numpy()

feature_name = test_data.columns

#test_X = test_data_numpyarray[:, : -1 ]
#test_y = test_data_numpyarray[:,-1]

test_X = test_data.iloc[:,:-1]
test_y = test_data.iloc[:,-1]

features_dataset1, features_dataset2, target_dataset1, target_dataset2 = train_test_split(test_X, test_y, train_size=0.5, random_state=8)


# First test whether the custom parameters are being assigned properly
def test_init_custom_parameters():
    """Assert that the AUTOQTL instantiator stores the AUTOQTL variables properly. """
    autoqtl_obj = AUTOQTLRegressor(
        population_size=500,
        generations=1000,
        offspring_size=2000,
        mutation_rate=0.05,
        crossover_rate=0.9,
        scoring='r2',
        verbosity=1,
        random_state=42,
        disable_update_check=True,
        warm_start=True,
        log_file=None
    )

    assert autoqtl_obj.population_size == 500
    assert autoqtl_obj.generations == 1000
    assert autoqtl_obj.offspring_size == 2000
    assert autoqtl_obj.mutation_rate == 0.05
    assert autoqtl_obj.crossover_rate == 0.9
    assert autoqtl_obj.scoring_function == 'r2'
    assert autoqtl_obj.max_time_mins is None
    assert autoqtl_obj.warm_start is True
    assert autoqtl_obj.verbosity == 1
    assert autoqtl_obj.log_file == None
    #print("No AssertError. Custom variables assigned successfully. ")

    autoqtl_obj._fit_init()

    assert autoqtl_obj._pop == []
    assert autoqtl_obj._pareto_front == None
    assert autoqtl_obj._last_optimized_pareto_front == None
    assert autoqtl_obj._last_optimized_pareto_front_n_gens == 0
    assert autoqtl_obj._optimized_pipeline == None
    assert autoqtl_obj._optimized_pipeline_score == None
    assert autoqtl_obj.fitted_pipeline_ == None
    assert autoqtl_obj._exported_pipeline_text == []
    assert autoqtl_obj.log_file_ == sys.stdout
    #print("No AssertError. Custom variables assigned successfully in _fit_init(). ")

# Test whether the log file is assigned to have the right file handler
def test_init_log_file():
    """Assert AUTOQTL has right file handler to save progress. """
    cachedir = mkdtemp()
    file_name = cachedir + "/progress.log"
    file_handle = open(file_name, "w")
    autoqtl_obj = AUTOQTLRegressor(log_file=file_handle)
    autoqtl_obj._fit_init()
    assert autoqtl_obj.log_file_ == file_handle
    file_handle.close()
    # clean up
    rmtree(cachedir)
    #print("No AssertError. AUTOQTL has right file handler to save progress. ")
    

# Test if fit() works to give a pipeline
def test_fit():
    """Assert that the AUTOQTL fit function provides an optimized pipeline. """
    autoqtl_obj = AUTOQTLRegressor(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=0
    )
    autoqtl_obj.fit(features_dataset1, target_dataset1, features_dataset2, target_dataset2)

    assert isinstance(autoqtl_obj._optimized_pipeline, creator.Individual)
    assert not (autoqtl_obj._start_datetime is None)


# Test if the optimized pipeline is being assigned properly
def test_update_top_pipeline():
    """Assert that the AUTOQTL _update_top_pipeline updated an optimized pipeline. """
    autoqtl_obj = AUTOQTLRegressor(
        random_state=42,
        population_size=1,
        offspring_size=2,
        generations=1,
        verbosity=1
    )
    autoqtl_obj.fit(features_dataset1, target_dataset1, features_dataset2, target_dataset2)
    autoqtl_obj._optimized_pipeline = None
    autoqtl_obj.fitted_pipeline_ = None
    autoqtl_obj._update_top_pipeline()

    assert isinstance(autoqtl_obj._optimized_pipeline, creator.Individual)


# Test if the summary of the pipeline is being printed properly along with the working of the fit function
def test_summary_of_best_pipeline():
    """Testing the summary_of_best_pipeline function. """
    autoqtl_obj = AUTOQTLRegressor(
        random_state=42,
        population_size=100,
        #offspring_size=2,
        generations=5,
        verbosity=3
    )
    autoqtl_obj.fit(features_dataset1, target_dataset1, features_dataset2, target_dataset2)
    #autoqtl_obj._summary_of_best_pipeline(features_dataset1, target_dataset2, features_dataset2, target_dataset2)
    assert isinstance(autoqtl_obj._optimized_pipeline, creator.Individual)
    #autoqtl_obj.get_feature_importance(features_dataset1, target_dataset1, random_state=0)
    autoqtl_obj.get_feature_importance(test_X, test_y, random_state=0)
    #autoqtl_obj.get_shap_values(features_dataset1, target_dataset1)

# Printing the Linear Regression R2 values for whole dataset and split dataset
def get_R2():
    model = LinearRegression()
    # Entire Dataset
    model.fit(test_X, test_y)
    print("Entire Dataset: ", model.score(test_X, test_y))
    # Dataset split 1
    model.fit(features_dataset1, target_dataset1)
    print("Dataset split 1: ", model.score(features_dataset1,target_dataset1))
    # Dataset split 1
    model.fit(features_dataset2, target_dataset2)
    print("Dataset split 2: ", model.score(features_dataset2,target_dataset2))

# calling the test functions
#test_init_custom_parameters()
#test_init_log_file()
#test_fit()
#test_update_top_pipeline()
test_summary_of_best_pipeline() # using 
get_R2()
#print(feature_name)
