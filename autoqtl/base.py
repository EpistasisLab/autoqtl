"""This file is part of AUTOQTL library"""
import numpy as np
import deap
from deap import base, creator, tools, gp

from sklearn.base import BaseEstimator

from .config.regressor import regressor_config_dict

"""Building up the initial GP. """

class AUTOQTLBase(BaseEstimator):
    """Automatically creates and optimizes machine learning pipelines using Genetic Programming"""