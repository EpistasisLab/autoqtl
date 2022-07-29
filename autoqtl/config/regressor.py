"""This python file contains the configuration dictionary with all the selctors, transformer and ML methods, 
which all will be converted to genetic programming operators to form the expression trees.

The custom AutoQTL configuration must be in nested dictionary format, 
where the first level key is the path and name of the operator and the 
second level key is the corresponding parameter name for that operator. 
The second level key should point to a list of parameter values for that parameter, i.e the hyperparameters. """

import numpy as np

regressor_config_dict = {
    
    # Feature Selectors
    'sklearn.feature_selection.SelectPercentile': {
        'percentile': [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95],
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

    'autoqtl.builtins.genetic_encoders.DominantEncoder': {

    },

    'autoqtl.builtins.genetic_encoders.RecessiveEncoder': {

    },

    'autoqtl.builtins.genetic_encoders.HeterosisEncoder': {

    },

    

    'autoqtl.builtins.genetic_encoders.UnderDominanceEncoder': {

    },

   
    'autoqtl.builtins.genetic_encoders.OverDominanceEncoder': {

    },

    

    # Machine Learning Methods

    'sklearn.linear_model.LinearRegression': {

    },

    'sklearn.tree.DecisionTreeRegressor': {
        'max_depth': range(1, 11),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21)
    },

    'sklearn.ensemble.RandomForestRegressor': {
        'n_estimators': [100],
        'max_features': np.arange(0.05, 1.01, 0.05),
        'min_samples_split': range(2, 21),
        'min_samples_leaf': range(1, 21),
        'bootstrap': [True, False]
    }

}









