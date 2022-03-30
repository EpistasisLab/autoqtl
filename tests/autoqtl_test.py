from shutil import rmtree
import sys
from tempfile import mkdtemp

from sklearn.model_selection import train_test_split
sys.path.append("C:/Users/ghosha/.vscode/autoqtl")
import autoqtl

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

# data
test_data = pd.read_csv("tests/randomset3.csv")
test_data_numpyarray = pd.DataFrame(test_data).to_numpy()

test_X = test_data_numpyarray[:, : -1 ]
test_y = test_data_numpyarray[:,-1]

features_dataset1, features_dataset2, target_dataset1, target_dataset2 = train_test_split(test_X, test_y, train_size=0.5, random_state=42)

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
    print("No AssertError. Custom variables assigned successfully. ")

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
    print("No AssertError. Custom variables assigned successfully in _fit_init(). ")

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
    print("No AssertError. AUTOQTL has right file handler to save progress. ")
    

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






# calling the test functions
test_init_custom_parameters()
test_init_log_file()
test_fit()
