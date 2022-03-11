"""This python file contains the configuration dictionary with all the selctors, transformer and ML methods, 
which all will be converted to genetic programming operators.

The custom AUTOQTL configuration must be in nested dictionary format, 
where the first level key is the path and name of the operator and the 
second level key is the corresponding parameter name for that operator. 
The second level key should point to a list of parameter values for that parameter. """

import numpy as np

regressor_config_dict = {
    
    # Feature Selectors
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': range(5, 95),
        'score_func': {
            'sklearn.feature_selection.f_regression': None
        }
    },

    'sklearn.feature_selection.VarianceThreshold': {
        'threshold': [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    },

    'autoqtl.builtins.feature_encoding_frequency_selector.FeatureEncodingFrequencySelector': {
        'threshold': [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35]
    },

    # Feature Transformers (Encoders)

    'autoqtl.builtins.genetic_encoders.AdditiveEncoder': {

    },

    'autoqtl.builtins.genetic_encoders.AdditiveAlternateEncoder': {

    },

    'autoqtl.builtins.genetic_encoders.DominantEncoder': {

    },

    'autoqtl.builtins.genetic_encoders.RecessiveEncoder': {

    },

    'autoqtl.builtins.genetic_encoders.HeterosisEncoder': {

    },

    # Machine Learning Methods

    'sklearn.linear_model.LinearRegression': {

    }
}









