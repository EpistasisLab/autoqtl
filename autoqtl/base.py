"""This file is part of AUTOQTL library. The AUTOQTLBase class is defined in this file. """
# all the import statements
from ast import expr
from datetime import date, datetime
import errno
from functools import partial
import imp
import inspect
import os
import random
from shutil import rmtree
from socket import timeout
import statistics
from subprocess import call
import sys
from tempfile import mkdtemp
import warnings
# from isort import file
from joblib import Memory
import numpy as np
import deap
from deap import base, creator, tools, gp
from pandas import DataFrame
import pandas as pd
# from pyrsistent import pset
from sklearn import tree
import re
import shap

import matplotlib.pyplot as plt
import seaborn as sns

from fpdf import FPDF


from sklearn.base import BaseEstimator
# from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import SCORERS, get_scorer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline, make_union, make_pipeline

from copy import copy, deepcopy

from sklearn.preprocessing import FunctionTransformer
from sklearn.utils import check_X_y, check_array, check_consistent_length
#from sympy import total_degree
from tqdm import tqdm

from .gp_types import Output_Array
#from .builtins.combine_dfs import CombineDFs
from .decorators import _pre_test
from .operator_utils import AUTOQTLOperatorClassFactory, Operator, ARGType

from .gp_deap import (
    cxOnePoint, get_feature_size, get_score_on_fitted_pipeline, mutNodeReplacement, _wrapped_score, eaMuPlusLambda, return_logbook
)

from .export_utils import (
    expr_to_tree,
    generate_pipeline_code,
    set_param_recursive,
    export_pipeline
)
#from .config.regressor import regressor_config_dict

try:
    from imblearn.pipeline import make_pipeline as make_imblearn_pipeline
except:
    make_imblearn_pipeline = None

"""Building up the initial GP. """

class AUTOQTLBase(BaseEstimator):
    """Automatically creates and optimizes machine learning pipelines using Genetic Programming. """
    
    regression = None # set to True by child class. Will be set to false in case of classification. We plan to include 'classification' functionalities in the future.

    def __init__(
        self,
        generations = 100,
        population_size = 100,
        offspring_size = None,
        mutation_rate = 0.9,
        crossover_rate = 0.1,
        scoring = None,
        subsample = 1.0,
        max_time_mins = None,
        max_eval_time_mins = 5,
        random_state = None,
        config_dict = None,
        template = None,
        warm_start = False,
        memory = None,
        periodic_checkpoint_folder = None,
        early_stop = None,
        verbosity = 0,
        log_file = None,
    ):
        """Setting up the genetic programming algorithm for pipeline optimization. All the parameters are initialized with the default values. 
        
        Parameters
        ----------
        generations: int or None, optional (default: 100)
            Number of iterations to the run pipeline optimization process.
            It must be a positive number or None. If None, the parameter
            max_time_mins must be defined as the runtime limit.
            Generally, AutoQTL will work better when you give it more generations (and
            therefore time) to optimize the pipeline but it also depends on the complexity of the data. AutoQTL will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        
        population_size: int, optional (default: 100)
            Number of individuals to retain in the GP population every generation.
            Generally, AutoQTL will work better when you give it more individuals
            (and therefore time) to optimize the pipeline. AUTOQTL will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        
        offspring_size: int, optional (default: None)
            Number of offspring to produce in each GP generation.
            By default, offspring_size = population_size.
        
        mutation_rate: float, optional (default: 0.9)
            Mutation rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the GP algorithm how many pipelines to apply random
            changes to every generation. We recommend using the default parameter unless
            you understand how the mutation rate affects GP algorithms.
        
        crossover_rate: float, optional (default: 0.1)
            Crossover rate for the genetic programming algorithm in the range [0.0, 1.0].
            This parameter tells the genetic programming algorithm how many pipelines to
            "breed" every generation. We recommend using the default parameter unless you
            understand how the mutation rate affects GP algorithms.
        
        scoring: string or callable, optional
            Function used to evaluate the quality of a given pipeline for the
            problem. By default, coeeficient of determination (r^2) is used for regression problems.
            Other built-in Regression metrics that can be used:
            ['neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error', 'r2']
        
        subsample: float, optional (default: 1.0)
            Subsample ratio of the training instance. Setting it to 0.5 means that AutoQTL
            randomly collects half of training samples for pipeline optimization process.
        
        
        max_time_mins: int, optional (default: None)
            How many minutes AutoQTL has to optimize the pipelines (basically how long will the generational process continue).
            If not None, this setting will allow AutoQTL to run until max_time_mins minutes
            elapsed and then stop. AutoQTL will stop earlier if generationsis set and all
            generations are already evaluated.
        
        max_eval_time_mins: float, optional (default: 5)
            How many minutes AutoQTL has to optimize a single pipeline.
            Setting this parameter to higher values will allow AutoQTL to explore more
            complex pipelines, but will also allow AutoQTL to run longer.
        
        random_state: int, optional (default: None)
            Random number generator seed for AutoQTL. Use this parameter to make sure
            that AutoQTL will give you the same results each time you run it against the
            same data set with that seed.
        
        config_dict: a Python dictionary or string, optional (default: None)
            Python dictionary:
                A dictionary customizing the operators and parameters that
                AutoQTL uses in the optimization process.
                For examples, see config_regressor.py 
            Path for configuration file:
                A path to a configuration file for customizing the operators and parameters that
                AutoQTL uses in the optimization process.
                For examples, see config_regressor.py 
        
        template: string (default: None)
            Template of predefined pipeline structure. The option is for specifying a desired structure
            for the machine learning pipeline evaluated in AutoQTL. So far this option only supports
            linear pipeline structure. Each step in the pipeline should be a main class of operators
            (Selector, Transformer or Regressor) or a specific operator
            (e.g. SelectPercentile) defined in AutoQTL operator configuration. If one step is a main class,
            AutoQTL will randomly assign all subclass operators (subclasses of SelectorMixin,
            TransformerMixin or RegressorMixin in scikit-learn) to that step.
            Steps in the template are delimited by "-", e.g. "SelectPercentile-Transformer-Regressor".
            By default value of template is None, AutoQTL generates tree-based pipeline randomly.
        
        warm_start: bool, optional (default: False)
            Flag indicating whether the AutoQTL instance will reuse the population from
            previous calls to fit().
        
        memory: a Memory object or string, optional (default: None)
            If supplied, pipeline will cache each transformer after calling fit. This feature
            is used to avoid computing the fit transformers within a pipeline if the parameters
            and input data are identical with another fitted pipeline during optimization process.
            String 'auto':
                AutoQTL uses memory caching with a temporary directory and cleans it up upon shutdown.
            String path of a caching directory
                AutoQTL uses memory caching with the provided directory and AutoQTL does NOT clean
                the caching directory up upon shutdown. If the directory does not exist, AutoQTL will
                create it.
            Memory object:
                AutoQTL uses the instance of joblib.Memory for memory caching,
                and AutoQTL does NOT clean the caching directory up upon shutdown.
            None:
                AutoQTL does not use memory caching.

        periodic_checkpoint_folder: path string, optional (default: None)
            If supplied, a folder in which AutoQTL will periodically save pipelines in pareto front so far while optimizing.
            Currently once per generation but not more often than once per 30 seconds.
            Useful in multiple cases:
                Sudden death before AutoQTL could save optimized pipeline
                Track its progress
                Grab pipelines while it's still optimizing
        
        early_stop: int or None (default: None)
            How many generations AutoQTL checks whether there is no improvement in optimization process.
            End optimization process if there is no improvement in the set number of generations.
        
        verbosity: int, optional (default: 0)
            How much information AutoQTL communicates while it's running.
            0 = none, 1 = minimal, 2 = high, 3 = all.
            A setting of 2 or higher will add a progress bar during the optimization procedure.
        
        log_file: string, io.TextIOWrapper or io.StringIO, optional (defaul: sys.stdout)
            Save progress content to a file.
        Returns
        -------
        None
        
        """
        if self.__class__.__name__ == "AUTOQTLBase":
            raise RuntimeError(
                "Do not instantiate the AUTOQTLBase class directly; use AUTOQTLRegressor instead."
            )
        
        self.population_size = population_size
        self.offspring_size = offspring_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.scoring = scoring
        self.subsample = subsample
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.early_stop = early_stop
        self.config_dict = config_dict
        self.template = template
        self.warm_start = warm_start
        self.memory = memory
        self.verbosity = verbosity
       
        self.random_state = random_state
        self.log_file = log_file

    def _setup_template(self, template):
        """Setup the template for the machine learning pipeline. 
        Accordingly set the minimum and maximum height of the initial GP trees. AutoQTL uses the default template, which is None.
        This allows the GP to generate diverse pipelines and explore more possible solutions. 
        
        Parameter
        ---------
        template : string
            template specifying the sequence of selectors/transformers/regressors to be used by designing ML pipelines by GP.
        
        Returns
        -------
        None
        """
        self.template = template
        if self.template is None:
            self._min = 1
            self._max = 3
        else:
            self._template_comp = template.split("-")
            self._min = 0
            self._max = 1
            for comp in self._template_comp:
                if comp == "CombineDFs":
                    self._min += 1
                    self._max += 2
                else:
                    self._min += 1
                    self._max += 1
        
        if self._max - self._min == 1:
            self.tree_structure = False
        else:
            self.tree_structure = True
    

    def _setup_scoring_function(self, scoring):
        """Setup the scoring function which will be used to score the pipelines generated.
        
        Parameter
        ---------
        scoring : string or callable function
            the custom scoring function specified by the user while using the package
            
        Returns
        -------
        None
        
        """
        if scoring:
            if isinstance(scoring, str):
                if scoring not in SCORERS:
                    raise ValueError(
                        "The scoring function {} is not available. "
                        "choose a valid scoring function from the AutoQTL " 
                        "documentation.".format(scoring)
                    )
                self.scoring_function = scoring 
            elif callable(scoring):
                # Heuristic to ensure user has not passed a metric
                module = getattr(scoring, "__module__", None)
                args_list = inspect.getfullargspec(scoring)[0]
                if args_list == ["y_true", "y_pred"] or (
                    hasattr(module, "startswith")
                    and (
                        module.startswith("sklearn.metrics.")
                        or module.startswith("autoqtl.metrics")
                    )
                    and not module.startswith("sklearn.metrics._scorer")
                    and not module.startswith("sklearn.metrics.tests.")
                ):
                    raise ValueError(
                        "Scoring function {} looks like it is a metric function "
                        "rather than a scikit-learn scorer. "
                        "Please update your custom scoring function.".format(scoring)
                    )
                else:
                    self.scoring_function = scoring
    
    def _setup_config(self, config_dict):
        """Setup the configuration dictionary containing the various ML methods, selectors and transformers.
        
        Parameters
        ----------
        config_dict : Python dictionary or string
            custom config dict containing the customizing operators and parameters that AutoQTL uses in the optimization process
            or the path to the cumtom config dict
            
        Returns
        -------
        None
        
        """
        
        if config_dict:
            if isinstance(config_dict, dict):
                self._config_dict = config_dict
            
            else:
                config = self._read_config_file(config_dict)
                if hasattr(config, "autoqtl_config"):
                    self._config_dict = config.autoqtl_config
                else:
                     raise ValueError(
                        'Could not find "autoqtl_config" in configuration file {}. '
                        "When using a custom config file for customizing operators "
                        "dictionary, the file must have a python dictionary with "
                        'the standardized name of "autoqtl_config"'.format(config_dict)
                    )
        else:
            self._config_dict = self.default_config_dict


    def _read_config_file(self, config_path):
        """Read the contents of the config dict file given the path.
        
        Parameters
        ----------
        config_path : string
            path to the config dictionary
        
        Returns
        -------
        None
        
        """
        if os.path.isfile(config_path):
            try:
                custom_config = imp.new_module("custom_config")

                with open(config_path, "r") as config_file:
                    file_string = config_file.read()
                    exec(file_string, custom_config.__dict__)
                return custom_config
            except Exception as e:
                raise ValueError(
                    "An error occured while attempting to read the specified "
                    "custom AutoQTL operator configuration file: {}".format(e)
                )
        else:
            raise ValueError(
                "Could not open specified AutoQTL operator config file: "
                "{}".format(config_path)
            )
    

    def _setup_pset(self):
        """Set up the Primitive set to contain the primitives (functions) which will be used to generate a strongly typed GP tree. 
            Uses the DEAP module class gp.PrimitiveSetTyped(...).
            
        """
        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._pset = gp.PrimitiveSetTyped("MAIN", [np.ndarray], Output_Array)
        self._pset.renameArguments(ARG0="input_matrix") # default names of the argument are ARG0 and ARG1, ARG0 is renamed to input_matrix
        self._add_operators() # function to add GP operators to the Primitive set for the GP tree to use them

        if self.verbosity > 2:
            print(
                "{} operators have been imported by AutoQTL.".format(len(self.operators))
            )

    def _add_operators(self):
        """Add the operators as primitives to the GP primitive set. The operators are in the form of python classes."""

        main_operator_types = ["Regressor", "Selector", "Transformer"] 
        return_types = [] 
        self._op_list = [] 

        if self.template == None: # default pipeline structure
            step_in_type = np.ndarray # Input type of each step/operator in the tree
            step_ret_type = Output_Array # Output type of each step/operator in the tree
            
            for operator in self.operators:
                arg_types = operator.parameter_types()[0][1:] # parameter_types() is defined in operator_utils.py, it returns the input and return types of an operator class
                if operator.root:
                    # In case an operator is a root, the return type of that operator can only be Output_Array. In AutoQTL, a ML method is always the root and cannot exist elsewhere in the tree.
                    tree_primitive_types = ([step_in_type] + arg_types, step_ret_type) # A tuple with input(arguments types) and output type of a root operator
                else:
                    # For a non-root operator the return type is n-dimensional array
                    tree_primitive_types = ([step_in_type] + arg_types, step_in_type)
                
                self._pset.addPrimitive(operator, *tree_primitive_types) # addPrimitive() is a method of the gp.PrimitiveSetTyped(...) class
                self._import_hash_and_add_terminals(operator, arg_types)
            #self._pset.addPrimitive(CombineDFs(), [step_in_type, step_in_type], step_in_type)

        else:
            gp_types = {}
            for idx, step in enumerate(self._template_comp):

                # input class in each step
                if idx:
                    step_in_type = return_types[-1]
                else:
                    step_in_type = np.ndarray
                if step != "CombineDFs":
                    if idx < len(self._template_comp) - 1:
                        # create an empty return class for returning class for strongly-type GP
                        step_ret_type_name = "Ret_{}".format(idx)
                        step_ret_type = type(step_ret_type_name, (object,), {})
                        return_types.append(step_ret_type)
                    else:
                        step_ret_type = Output_Array
                
                if step == "CombineDFs":
                    """self._pset.addPrimitive(
                        CombineDFs(), [step_in_type, step_in_type], step_in_type
                    )"""
                    pass
                elif main_operator_types.count(step): # if the step is a main type
                    step_operator_list = [op for op in self.operators if op.type() == step]
                    for operator in step_operator_list:
                        arg_types = operator.parameter_types()[0][1:]
                        p_types = ([step_in_type] + arg_types, step_ret_type)
                        self._pset.addPrimitive(operator, *p_types)
                        self._import_hash_and_add_terminals(operator, arg_types)
                else:  # if the step is a specific operator or a wrong input
                    try:
                        operator = next(
                            op for op in self.operators if op.__name__ == step
                        )
                    except:
                        raise ValueError(
                            "An error occured while attempting to read the specified "
                            "template. Please check a step named {}".format(step)
                        )
                    arg_types = operator.parameter_types()[0][1:]
                    p_types = ([step_in_type] + arg_types, step_ret_type)
                    self._pset.addPrimitive(operator, *p_types)
                    self._import_hash_and_add_terminals(operator, arg_types)
        self.return_types = [np.ndarray, Output_Array] + return_types 


    def _import_hash_and_add_terminals(self, operator, arg_types):
        """Call the _import_hash and _add_terminal methods """
        if not self._op_list.count(operator.__name__):
            self._import_hash(operator)
            self._add_terminals(arg_types)
            self._op_list.append(operator.__name__)
    

    def _import_hash(self, operator):
        """Import required modules into local namespace so that pipelines may be evaluated directly """
        for key in sorted(operator.import_hash.keys()): # import_hash is a dict containing the import paths for the classes, declared and defined in operator_utils.py
            module_list = ", ".join(sorted(operator.import_hash[key]))

            if key.startswith("autoqtl."):
                exec("from {} import {}".format(key[7:], module_list))
            else:
                exec("from {} import {}".format(key, module_list))

            for var in operator.import_hash[key]:
                self.operators_context[var] = eval(var)
    

    def _add_terminals(self, arg_types):
        """Adding terminals"""
        for _type in arg_types:
            type_values = list(_type.values) # picks up the multiple possible values for that argument. values is a property of the argument classes created

            for val in type_values:
                terminal_name = _type.__name__ + "=" + str(val)
                self._pset.addTerminal(val, _type, name=terminal_name)


    def _setup_toolbox(self):
        """Setup the toolbox. ToolBox is a DEAP package class, which is a toolbox for evolution containing all the evolutionary operators. """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            creator.create("FitnessMulti", base.Fitness, weights=(1.0, 1.0)) # Weights set according to requirement of maximizing two objectives (Test R^2 and difference score)
            creator.create(
                "Individual",
                gp.PrimitiveTree,
                fitness=creator.FitnessMulti,
                statistics=dict
            )

            self._toolbox = base.Toolbox()
            self._toolbox.register(
                "expr", self._gen_grow_safe, pset=self._pset, min_=self._min, max_=self._max
            )
            self._toolbox.register(
                "individual", tools.initIterate, creator.Individual, self._toolbox.expr
            )
            self._toolbox.register(
            "population", tools.initRepeat, list, self._toolbox.individual
            )
            self._toolbox.register("compile", self._compile_to_sklearn)
            self._toolbox.register("select", tools.selNSGA2)
            self._toolbox.register("mate", self._mate_operator)
            if self.tree_structure:
                self._toolbox.register(
                "expr_mut", self._gen_grow_safe, min_=self._min, max_=self._max + 1
                )
            else:
                self._toolbox.register(
                "expr_mut", self._gen_grow_safe, min_=self._min, max_=self._max
                )
            self._toolbox.register("mutate", self._random_mutation_operator)
    

    def _gen_grow_safe(self, pset, min_, max_, type_=None):
        """Generate an expression where each leaf might have a different depth between min_ and max_.
        Parameters
        ----------
        pset: PrimitiveSetTyped
            Primitive set from which primitives are selected.
        min_: int
            Minimum height of the produced trees.
        max_: int
            Maximum Height of the produced trees.
        type_: class
            The type that should return the tree when called, when
                  :obj:None (default) the type of :pset: (pset.ret)
                  is assumed.
        Returns
        -------
        individual: list
            A grown tree with leaves at possibly different depths.
        """

        def condition(height, depth, type_):
            """Stop when the depth is equal to height or when a node should be a terminal."""
            return type_ not in self.return_types or depth == height

        return self._generate(pset, min_, max_, condition, type_)


    @_pre_test
    def _generate(self, pset, min_, max_, condition, type_=None):
        """Generate a Tree as a list of lists.
        The tree is build from the root to the leaves, and it stop growing when
        the condition is fulfilled.
        Parameters
        ----------
        pset: PrimitiveSetTyped
            Primitive set from which primitives are selected.
        min_: int
            Minimum height of the produced trees.
        max_: int
            Maximum height of the produced trees.
        condition: function
            The condition is a function that takes two arguments,
            the height of the tree to build and the current
            depth in the tree.
        type_: class
            The type that should return the tree when called, when
            :obj:None (default) no return type is enforced.
        Returns
        -------
        individual: list
            A grown tree with leaves at possibly different depths
            depending on the condition function.
        """
        if type_ is None:
            type_ = pset.ret
        expr = []
        height = np.random.randint(min_, max_)
        stack = [(0, type_)]
        while len(stack) != 0:
            depth, type_ = stack.pop()

            # We've added a type_ parameter to the condition function
            if condition(height, depth, type_):
                try:
                    term = np.random.choice(pset.terminals[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError(
                        "The gp.generate function tried to add "
                        "a terminal of type {}, but there is"
                        "none available. {}".format(type_, traceback)
                    )
                if inspect.isclass(term):
                    term = term()
                expr.append(term)
            else:
                try:
                    prim = np.random.choice(pset.primitives[type_])
                except IndexError:
                    _, _, traceback = sys.exc_info()
                    raise IndexError(
                        "The gp.generate function tried to add "
                        "a primitive of type {}, but there is"
                        "none available. {}".format(type_, traceback)
                    )
                expr.append(prim)
                for arg in reversed(prim.args):
                    stack.append((depth + 1, arg))
        return expr


    @_pre_test
    def _mate_operator(self, ind1, ind2):
        """Crossover operator, one point crossover is used (DEAP inbuilt tool cxOnePoint is modified according to the problem).
        Parameters
        ----------
        ind1 : DEAP individual
            A list of pipeline operators and model parameters that can be compiled by DEAP into a callable function
        
        ind2 : DEAP individual
            A list of pipeline operators and model parameters that can be compiled by DEAP into a callable function
        
        Returns
        -------
        offspring : DEAP individual formed after crossover
        
        offspring2 : DEAP individual formed after crossover
        
        """
        for _ in range(self._max_mut_loops):
            ind1_copy, ind2_copy = self._toolbox.clone(ind1), self._toolbox.clone(ind2)
            offspring, offspring2 = cxOnePoint(ind1_copy, ind2_copy)

            if str(offspring) not in self.evaluated_individuals_:
                # We only use the first offspring, so we do not care to check uniqueness of the second.

                # update statistics:
                # mutation_count is set equal to the sum of mutation_count's of the predecessors
                # crossover_count is set equal to the sum of the crossover_counts of the predecessor +1, corresponding to the current crossover operations
                # predecessor is taken as tuple string representation of two predecessor individuals
                # generation is set to 'INVALID' such that we can recognize that it should be updated accordingly
                offspring.statistics["predecessor"] = (str(ind1), str(ind2))
               
                offspring.statistics["mutation_count"] = (
                    ind1.statistics["mutation_count"]
                    + ind2.statistics["mutation_count"]
                )

                offspring.statistics["crossover_count"] = (
                    ind1.statistics["crossover_count"]
                    + ind2.statistics["crossover_count"]
                    + 1
                )

                offspring.statistics["generation"] = "INVALID"
                break
        
        return offspring, offspring2


    @_pre_test
    def _random_mutation_operator(self, individual, allow_shrink=True):
        """Perform a replacement, insertion, or shrink mutation on an individual.
        Parameters
        ----------
        individual: DEAP individual
            A list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function
        allow_shrink: bool (True)
            If True the `mutShrink` operator, which randomly shrinks the pipeline,
            is allowed to be chosen as one of the random mutation operators.
            If False, `mutShrink`  will never be chosen as a mutation operator.
        Returns
        -------
        mut_ind: DEAP individual
            Returns the individual with one of the mutations applied to it
        """
        if self.tree_structure:
            mutation_techniques = [
                partial(gp.mutInsert, pset=self._pset),
                partial(mutNodeReplacement, pset=self._pset),
            ]
            # We can't shrink pipelines with only one primitive, so we only add it if we find more primitives.
            number_of_primitives = sum(
                isinstance(node, deap.gp.Primitive) for node in individual
            )
            if number_of_primitives > 1 and allow_shrink:
                mutation_techniques.append(partial(gp.mutShrink))
        else:
            mutation_techniques = [partial(mutNodeReplacement, pset=self._pset)]

        mutator = np.random.choice(mutation_techniques)

        unsuccesful_mutations = 0
        for _ in range(self._max_mut_loops):
            # We have to clone the individual because mutator operators work in-place.
            ind = self._toolbox.clone(individual)
            (offspring,) = mutator(ind)
            if str(offspring) not in self.evaluated_individuals_:
                # Update statistics
                # crossover_count is kept the same as for the predecessor
                # mutation count is increased by 1
                # predecessor is set to the string representation of the individual before mutation
                # generation is set to 'INVALID' such that we can recognize that it should be updated accordingly
                offspring.statistics["crossover_count"] = individual.statistics[
                    "crossover_count"
                ]
                offspring.statistics["mutation_count"] = (
                    individual.statistics["mutation_count"] + 1
                )
                offspring.statistics["predecessor"] = (str(individual),)
                offspring.statistics["generation"] = "INVALID"
                break
            else:
                unsuccesful_mutations += 1
        # Sometimes you have pipelines for which every shrunk version has already been explored too.
        # To still mutate the individual, one of the two other mutators should be applied instead.
        if (unsuccesful_mutations == 50) and (
            type(mutator) is partial and mutator.func is gp.mutShrink
        ):
            (offspring,) = self._random_mutation_operator(
                individual, allow_shrink=False
            )

        return (offspring,)

    
    def _compile_to_sklearn(self, expr):
        """Compile a DEAP pipeline into a sklearn pipeline.
        Parameters
        ----------
        expr: DEAP individual
            The DEAP pipeline to be compiled
        Returns
        -------
        sklearn_pipeline: sklearn.pipeline.Pipeline
        """
        sklearn_pipeline_str = generate_pipeline_code(
            expr_to_tree(expr, self._pset), self.operators
        )
        sklearn_pipeline = eval(sklearn_pipeline_str, self.operators_context)
        sklearn_pipeline.memory = self._memory
        if self.random_state:
            # Fix random state when the operator allows
            set_param_recursive(
                sklearn_pipeline.steps, "random_state", self.random_state
            )
      
        return sklearn_pipeline
    

    def _get_make_pipeline_func(self):
        """Utility function to make a sklearn pipeline. The function to make the pipeline is choosen according to the presence of resamplers in the config dict."""
        imblearn_used = np.any([k.count("imblearn") for k in self._config_dict.keys()])

        if imblearn_used == True:
            assert make_imblearn_pipeline is not None, "You must install `imblearn`"
            make_pipeline_func = make_imblearn_pipeline
        else:
            make_pipeline_func = make_pipeline

        return make_pipeline_func


    def _preprocess_individuals(self, individuals):
        """Preprocess DEAP individuals before pipeline evaluation.
         Parameters
        ----------
        individuals: a list of DEAP individual
            One individual is a list of pipeline operators and model parameters that can be
            compiled by DEAP into a callable function
        Returns
        -------
        operator_counts: dictionary
            a dictionary of operator counts in individuals for evaluation
        eval_individuals_str: list
            a list of string of individuals for evaluation
        sklearn_pipeline_list: list
            a list of scikit-learn pipelines converted from DEAP individuals for evaluation
        stats_dicts: dictionary
            A dict where 'key' is the string representation of an individual and 'value' is a dict containing statistics about the individual
        """
        # update self._pbar.total
        if (
            not (self.max_time_mins is None)
            and not self._pbar.disable
            and self._pbar.total <= self._pbar.n
        ):
            self._pbar.total += self._lambda
        
        # check we do not evaluate twice the same individual in one pass.
        _, unique_individual_indices = np.unique(
            [str(ind) for ind in individuals], return_index=True
        )

        unique_individuals = [
            ind for i, ind in enumerate(individuals) if i in unique_individual_indices
        ]
        # update number of duplicate pipelines, the progress bar will show many pipleines have been processed
        self._update_pbar(pbar_num = len(individuals) - len(unique_individuals))

        # a dictionary for storing operator counts of an individual(pipeline)
        operator_counts = {}

        # a dictionary for storing all the statistics of an individual(pipleine)
        stats_dicts = {}

        # 2 lists of DEAP individuals' one in string format and their corresponding sklearn pipeline for parallel computing
        eval_individuals_str = []
        sklearn_pipeline_list = []

        for individual in unique_individuals:
            # Disallow certain combinations of operators because they will take too long or take up too much RAM
            individual_str = str(individual)
            if not len(individual): # a pipeline cannot be randomly generated
                self.evaluated_individuals_[
                    individual_str
                ] = self._combine_individual_stats(
                    5000.0, -float("inf"), -float("inf"), individual.statistics
                ) # randomly set 5000, to make the pipeline not a suitable pipeline, needs to be changed later
                self._update_pbar(
                    pbar_msg = "Invalid pipeline encountered. Skipping its evaluation."
                ) 
                continue
            sklearn_pipeline_str = generate_pipeline_code(expr_to_tree(individual, self._pset), self.operators)
            if sklearn_pipeline_str.count("PolynomialFeatures") > 1:
                self.evaluated_individuals_[
                    individual_str
                ] = self._combine_individual_stats(
                    5000.0, -float("inf"), -float("inf"), individual.statistics
                )   
                self._update_pbar(
                    pbar_msg = "Invalid pipeline encountered. Skipping its evaluation."
                )
            # Check if the individual was evaluated before
            elif individual_str in self.evaluated_individuals_:
                """self._update_pbar(
                    pbar_msg=(
                        "Pipeline encountered that has previously been evaluated during the "
                        "optimization process. Using the score from the previous evaluation. "
                    )
                )"""
                pass
                
            else:
                try:
                    # Transform the tree expression into an sklearn pipeline
                    sklearn_pipeline = self._toolbox.compile(expr=individual)

                    # Count the number of pipeline operators as a measure of pipeline complexity, autoqtl does not use this info at the moment but might need it in future
                    operator_count = self._operator_count(individual)
                    operator_counts[individual_str] = max(1, operator_count)

                    stats_dicts[individual_str] = individual.statistics
                except Exception:
                    self.evaluated_individuals_[
                        individual_str
                    ] = self._combine_individual_stats(
                        5000.0, -float("inf"), -float("inf"), individual.statistics
                    )
                    self._update_pbar()
                    continue
                eval_individuals_str.append(individual_str)
                sklearn_pipeline_list.append(sklearn_pipeline)
            
        return operator_counts, eval_individuals_str, sklearn_pipeline_list, stats_dicts
    

    def _operator_count(self, individual):
        """Count the number of pipeline operators as a measure of pipeline complexity.
        Parameters
        ----------
        individual: list
            A grown tree with leaves at possibly different depths
            depending on the condition function.
        Returns
        -------
        operator_count: int
            How many operators in a pipeline
        """
        operator_count = 0
        for node in individual:
            if type(node) is deap.gp.Primitive and node.name != "CombineDFs":
                operator_count += 1
        return operator_count


    def _combine_individual_stats(self, operator_count, test_R2, difference_score ,individual_stats):
        """Combine the stats with operator count and cv score and preprare to be written to _evaluated_individuals
        Parameters
        ----------
        operator_count: int
            number of components in the pipeline
        test_R2: float
            internal score assigned to the pipeline by the evaluate operator on dataset2 (test) by training a model on daatset1, basically the R2 score
        difference_score: float
            a score to penalize training/testing difference
        individual_stats: dictionary
            dict containing statistics about the individual. currently:
            'generation': generation in which the individual was evaluated
            'mutation_count': number of mutation operations applied to the individual and its predecessor cumulatively
            'crossover_count': number of crossover operations applied to the individual and its predecessor cumulatively
            'predecessor': string representation of the individual
        Returns
        -------
        stats: dictionary
            dict containing the combined statistics:
            'operator_count': number of operators in the pipeline
            'test_R^2': internal score assigned to the pipeline, basically the R2 score on daatset2(test)
            'difference_score': training/testing difference score
            and all the statistics contained in the 'individual_stats' parameter
        """
        stats = deepcopy(
            individual_stats
        )  # Deepcopy, since the string reference to predecessor should be cloned
        stats["operator_count"] = operator_count
        stats["test_R^2"] = test_R2
        stats["difference_score"] = difference_score
        return stats # returns the entire statistics dictionary of the pipeline with all the components
    

    def _evaluate_individuals(
        self, population, features_dataset1, target_dataset1, features_dataset2, target_dataset2, sample_weight=None
    ):
        """Determine the fit of the provided individuals. Evaluate each pipeline and return the fitness scores.
        
        Parameters
        ----------
        population : a list of DEAP individual
            One individual is a list of pipeline operators and model parameters that can be compiled by DEAP into a callable function
        features_dataset1 : numpy.ndarray {n_samples, n_features}
            A numpy matrix containing the training and testing features for the individual's evaluation. A part of dataset1 for evaluation
        target_dataset1 : numpy.ndarray {n_samples}
            A numpy matrix containing the training and testing target for the individual's evaluation
        features_dataset2 : numpy.ndarray {n_samples, n_features}
            A numpy matrix containing the training and testing features for the individual's evaluation. A part of dataset1 for evaluation
        target_dataset2 : numpy.ndarray {n_samples}
            A numpy matrix containing the training and testing target for the individual's evaluation
        sample_weight: array-like {n_samples}, optional
            List of sample weights to balance (or un-balanace) the dataset target as needed
            
        Returns
        -------
        fitnesses_ordered: float
            Returns a list of tuple value indicating the individual's fitness
            according to its performance on the provided data
            
        """
        # Evaluate the individuals with an invalid fitness
        individuals = [ind for ind in population if not ind.fitness.valid]
        num_population = len(population)
        # update pbar for valid individuals (valid individuals have fitness values) 
        if self.verbosity > 0:
            self._pbar.update(num_population - len(individuals))

        # preprocess the individuals with invalid fitness
        (
            operator_counts,
            eval_individuals_str,
            sklearn_pipeline_list,
            stats_dicts,
        ) = self._preprocess_individuals(individuals)

        partial_wrapped_score = partial(
            _wrapped_score,
            scoring_function = self.scoring_function,
            sample_weight = sample_weight,
            timeout=max(int(self.max_eval_time_mins * 60), 1)
        ) # The values for sklearn pipeline and (features, target) will change in every function call

        result_score_list = []

        try:
            # check time limit before pipeline evaluation
            self._stop_by_max_time_mins()

            # check for parallelization, AutoQTL does not use parallelization now
            # Training/testing R2 values are calculated for each individual pipelines.

            for sklearn_pipeline in sklearn_pipeline_list:
                self._stop_by_max_time_mins()
                score_on_dataset1 = partial_wrapped_score(sklearn_pipeline=sklearn_pipeline, features=features_dataset1, target=target_dataset1)
                #score_on_dataset1 = get_score_on_fitted_pipeline(sklearn_pipeline=sklearn_pipeline, X_learner=features_dataset2, y_learner=target_dataset2, X_test=features_dataset1, y_test=target_dataset1, scoring_function=self.scoring_function)
                
                #score_on_dataset2 = partial_wrapped_score(sklearn_pipeline=sklearn_pipeline, features=features_dataset2, target=target_dataset2)
                score_on_dataset2 = get_score_on_fitted_pipeline(sklearn_pipeline=sklearn_pipeline, X_learner=features_dataset1, y_learner=target_dataset1, X_test=features_dataset2, y_test=target_dataset2, scoring_function=self.scoring_function)
                difference_score = (1/(abs(score_on_dataset1-score_on_dataset2)))**(1/4)
                result_score_list = self._update_val(score_on_dataset2, difference_score, result_score_list)
                
                test_score = _wrapped_score(sklearn_pipeline, features_dataset1, target_dataset1, self.scoring_function, sample_weight, timeout=max(int(self.max_eval_time_mins*60), 1))
                
                

        except (KeyboardInterrupt, SystemExit, StopIteration) as e:
            if self.verbosity > 0:
                self._pbar.write("", file=self.log_file_)
                self._pbar.write(
                    "{}\nAutoQTL closed during evaluation in one generation.\n"
                    "WARNING: AutoQTL may not provide a good pipeline if AutoQTL is stopped/interrupted in a early generation.".format(
                        e
                    ),
                    file=self.log_file_,
                )
            
            # number of individuals already evaluated in this generation, update those individuals
            num_eval_ind = len(result_score_list)
            self._update_evaluated_individuals_(
                result_score_list,
                eval_individuals_str[:num_eval_ind],
                operator_counts,
                stats_dicts,
            )
            for ind in individuals[:num_eval_ind]:
                ind_str = str(ind)
                ind.fitness.values = (
                    self.evaluated_individuals_[ind_str]["test_R^2"],
                    self.evaluated_individuals_[ind_str]["difference_score"]
                ) # evaluated_individuals_ is a dictionary containing the evaluated individuals in the previous generations, defined in the fit_init() function

            self._pareto_front.update(individuals[:num_eval_ind]) # the update() is the inbuilt function of pareto front of DEAP

            self._pop = population
            raise KeyboardInterrupt 

        self._update_evaluated_individuals_(
            result_score_list, eval_individuals_str, operator_counts, stats_dicts
        )    # update when no error/interruption occurred, so all the recently evaluated individuals get updated

        for ind in individuals:
            ind_str = str(ind)
            ind.fitness.values = (
                self.evaluated_individuals_[ind_str]["test_R^2"],
                self.evaluated_individuals_[ind_str]["difference_score"],
            )
        individuals = [ind for ind in population if not ind.fitness.valid] # WHY IS THIS DONE? Contains the new list of individuals with invalid scores
        self._pareto_front.update(population)

        return population # returns the population and sets the fitness score of the evaluated individuals

    # Function to update the progress bar(instance of tqdm)
    def _update_pbar(self, pbar_num=1, pbar_msg=None):
        """Update self._pbar and error message during pipeline evaluation.
        
        Parameters
        ----------
        pbar_num : int
            How many pipelines has been processed
        pbar_msg : None or string
            Error message
            
        Returns
        -------
        None
        
        """
        if not isinstance(self._pbar, type(None)):
            if self.verbosity > 2 and pbar_msg is not None:
                self._pbar.write(pbar_msg, file=self.log_file) 
            if not self._pbar.disable:
                self._pbar.update(pbar_num)

    # Function to update the two calculated scores for the pipleine in the list of result scores and update self._pbar during pipeline evaluation. 
    def _update_val(self, score1, score2, result_score_list):
        """Update the score of the pipeline evaluation on the two datasets d1 and d2 in the result score list and update self._pbar to show the total number of pipelines proccessed
        
        Parameters
        ----------
        score1 : float or "Timeout"
            Test R^2
        score2 : float or "Timeout"
            Difference Score
        result_score_list : list
            A list of scores of the pipelines [a list of list]
        Returns
        -------
        result_score_list : list
            An updated result score list
        """
        self._update_pbar()
        score_list = []
        if score1 == "Timeout" or score2 == "Timeout" : # if any of the pipeline scores on any dataset is "Timeout" invalidate the score of the pipeline
            self._update_pbar(
                pbar_msg=(
                    "Skipped pipeline #{0} due to time out. "
                    "Continuing to the next pipeline.".format(self._pbar.n)
                )
            )
            score_list.append(-float("inf")) # score1 invalidated
            score_list.append(-float("inf")) # score2 invalidated
        else:
            score_list.append(score1)
            score_list.append(score2)
        
        result_score_list.append(score_list)

        return result_score_list

    # Function to update the invalid individuals evaluated in the most recent call
    def _update_evaluated_individuals_(
        self, result_score_list, eval_individuals_str, operator_counts, stats_dicts
    ):
        """Update self.evaluated_individuals_ (dict storing the evaluated individuals from the prev gens) and error message during pipeline evaluation.
        
        Parameters
        ----------
        result_score_list : list
            A list of scores for evaluated pipelines. Basically it is a list of list, with two R2 values on the two datasets
        eval_individuals_str : list
            A list of strings for evaluated pipelines
        operator_counts : dict
            A dict where 'key' is the string representation of an individual and 'value' is the number of operators in the pipeline
        stats_dict : dict
            A dict where 'key' is the string representation of an individual and 'value' is a dict containing statistics about the individual
            
        Returns
        -------
        None
        
        """
        for result_score, individual_str in zip(
            result_score_list, eval_individuals_str
        ):
            if type(result_score[0]) in [float, np.float64, np.float32] and type(result_score[1]) in [float, np.float64, np.float32] :
                self.evaluated_individuals_[
                    individual_str
                ] = self._combine_individual_stats(
                    operator_counts[individual_str],
                    result_score[0],
                    result_score[1],
                    stats_dicts[individual_str],
                )
            else:
                raise ValueError("Scoring function does not return a float.")

    # function check time limit of optimization
    def _stop_by_max_time_mins(self):
        """Stop optimization process once maximum minutes have elapsed. """
        if self.max_time_mins:
            total_mins_elapsed = (
                datetime.now() - self._start_datetime
            ).total_seconds() / 60.0
            if total_mins_elapsed >= self.max_time_mins:
                raise KeyboardInterrupt(
                    "{:.2f} minutes have elapsed. AUTOQTL will close down.".format(
                        total_mins_elapsed
                    )
                )
            
    # update the best pipeline generated till present generation
    def _update_top_pipeline(self):
        """Helper function to update the _optimized_pipeline(will store the best pipleine) field. """
        # Store the pipeline with the highest internal testing score
        if self._pareto_front:
            #print("Pareto Front formed") # trying to debug
            #print(self._pareto_front.items) # trying to debug
            #print(self._pareto_front.keys) # trying to debug
            self._optimized_pipeline_score = [-float("inf"), -float("inf")] # We will store the pipeline test R^2 and difference score as the final score
            for pipeline, pipeline_scores in zip(
                self._pareto_front.items, reversed(self._pareto_front.keys) # pipeline_score picks up the fitness value tuple in the list of keys
            ):
                if (pipeline_scores.wvalues[0] > self._optimized_pipeline_score[0]) and (pipeline_scores.wvalues[1] > self._optimized_pipeline_score[1]): 
                    self._optimized_pipeline = pipeline
                    self._optimized_pipeline_score = [pipeline_scores.wvalues[0], pipeline_scores.wvalues[1]]
                    #print(pipeline) # trying to debug
            if not self._optimized_pipeline : # Did not find any best optimized pipeline
                # pick one individual from evaluated pipeline for an error message
                eval_ind_list = list(self.evaluated_individuals_.keys())
                for pipeline, pipeline_scores in zip(
                    self._pareto_front.items, reversed(self._pareto_front.keys)
                ):
                    if np.isinf(pipeline_scores.wvalues[0]) or np.isinf(pipeline_scores.wvalues[1]):
                        sklearn_pipeline = self._toolbox.compile(expr=pipeline)
                        break # TPOT calculated the cross validation score
                raise RuntimeError(
                    "There was an error in the AutoQTL optimization process. Please make sure you passed the data to AutoQTL correctly."
                )
            else:
                pareto_front_wvalues = [
                    [pipeline_scores.wvalues[0], pipeline_scores.wvalues[1]]
                    for pipeline_scores in self._pareto_front.keys
                ]
                if not self._last_optimized_pareto_front:
                    self._last_optimized_pareto_front = pareto_front_wvalues
                elif self._last_optimized_pareto_front == pareto_front_wvalues:
                    self._last_optimized_pareto_front_n_gens +=1
                else:
                    self._last_optimized_pareto_front = pareto_front_wvalues
                    self._last_optimized_pareto_front_n_gens = 0
        else:
            # If user passes CTRL+C in initial generation, self._pareto_front(halloffame) should be not updated yet.
            # need raise RuntimeError because no pipeline has been optimized
            raise RuntimeError(
                "A pipeline has not yet been optimized. Please call fit() first. "
            )

    # check for an optimized pipeline
    def _check_periodic_pipeline(self, gen):
        """If enough time has passed, save a new optimized pipeline. Currently used in the per generation hook in the optimization loop.
        
        Parameters
        ----------
        gen : int
            Generation number
            
        Returns
        -------
        None
        
        """
        self._update_top_pipeline
        if self.periodic_checkpoint_folder is not None:
            total_since_last_pipeline_save = (
                datetime.now() - self._last_pipeline_write
            ).total_seconds()
            if(
                total_since_last_pipeline_save > self._output_best_pipeline_period_seconds
            ): # variable _output_best_pipeline_period_seconds is set to 30, so periodic pipelines doesn't get saved more often than this
                self._last_pipeline_write = datetime.now()
                self._save_periodic_pipeline(gen)
        
        if self.early_stop is not None:
            if self._last_optimized_pareto_front_n_gens >= self.early_stop:
                raise StopIteration(
                    "The optimized pipeline was not improved after evaluating {} more generations. "
                    "Will end the optimization process.\n".format(self.early_stop)
                )

    # Utility function to save a pipeline 
    def _save_periodic_pipeline(self, gen):
        """Saves the most optimized pipeline periodically. """
        try:
            self._create_periodic_checkpoint_folder()
            for pipeline, pipeline_scores in zip(
                self._pareto_front.items, reversed(self._pareto_front.keys)
            ):
                idx = self._pareto_front.items.index(pipeline)
                pareto_front_pipeline_score = [pipeline_scores.wvalues[0], pipeline_scores.wvalues[1]]
                sklearn_pipeline_str = generate_pipeline_code(
                    expr_to_tree(pipeline, self._pset), self.operators
                ) # get the string format of the sklearn pipeline
                
                to_write = export_pipeline(
                    pipeline,
                    self.operators,
                    self._pset,
                    self._imputed,
                    pareto_front_pipeline_score,
                    self.random_state
                ) # call to function to generate the source code for the pipeline (Note: The pipeline score is in a list)

                # don't export a pipeline already exported earlier
                if self._exported_pipeline_text.count(sklearn_pipeline_str):
                    self._update_pbar(
                        pbar_num=0,
                        pbar_msg="Periodic pipeline was not saved, probably saved before..."
                    )
                else:
                    filename = os.path.join(
                        self.periodic_checkpoint_folder,
                        "pipeline_gen_{}_idx_{}_{}.py".format(
                            gen, idx, datetime.now().strftime("%Y.%m.%d_%H-%M-%S")
                        ),
                    )
                    self._update_pbar(
                        pbar_num=0,
                        pbar_msg="Saving periodic pipeline from pareto front to {}".format(
                            filename
                        ),
                    )
                    with open(filename, "w") as output_file:
                        output_file.write(to_write)
                    self._exported_pipeline_text.append(sklearn_pipeline_str)

        except Exception as e:
            self._update_pbar(
                pbar_num=0,
                pbar_msg="Failed saving periodic pipeline, exception:\n{}".format(
                    str(e)[:250]
                ),
            )

    # Utility function to create a periodic checkpoint folder to  save periodic pipeline
    def _create_periodic_checkpoint_folder(self):
        """Creates a folder to store pipelines at periodic intervals. """
        try:
            os.makedirs(self.periodic_checkpoint_folder)
            self._update_pbar(
                pbar_msg="Create new folder to save periodic pipeline: {}".format(
                    self.periodic_checkpoint_folder
                )
            )
        except OSError as e:
            if e.errno == errno.EEXIST and os.path.isdir(
                self.periodic_checkpoint_folder
            ): # errno.EEXIST checks for file exist error
                pass # Folder already exists
            else:
                raise ValueError(
                    "Failed creating the periodic_checkpoint_folder:\n{}".format(e)
                )


    def export(self, output_file_name="", data_file_path=""):
        """Export the optimized pipeline as Python code.
        Parameters
        ----------
        output_file_name: string (default: '')
            String containing the path and file name of the desired output file. If left empty, writing to file will be skipped.
        data_file_path: string (default: '')
            By default, the path of input dataset is 'PATH/TO/DATA/FILE' by default.
            If data_file_path is another string, the path will be replaced.
        Returns
        -------
        to_write: str
            The whole pipeline text as a string.
        """
        if self._optimized_pipeline is None:
            raise RuntimeError(
                "A pipeline has not yet been optimized. Please call fit() first."
            )

        to_write = export_pipeline(
            self._optimized_pipeline,
            self.operators,
            self._pset,
            self._imputed,
            self._optimized_pipeline_score,
            self.random_state,
            data_file_path=data_file_path,
        )

        if output_file_name != "":
            with open(output_file_name, "w") as output_file:
                output_file.write(to_write)
        else:
            return to_write

    
    # Function to print out the best pipeline (on basis of two objectives) and also display the entire pareto front to the user
    def _summary_of_best_pipeline(self, features_dataset1, target_dataset1, features_dataset2, target_dataset2):
        """Print the best pipeline at the end of optimization process, using two datasets. 
        
        Parameters
        ----------
        features_dataset1 : array-like {n_samples, n_features}
            Feature matrix of Dataset1
        
        target_dataset1 : array-like {n_samples}
            Target values of Dataset1
            
        features_dataset2 : array-like {n_samples, n_features}
            Feature matrix of Dataset2
        
        target_dataset2 : array-like {n_samples}
            Target values of Dataset2
            
        Returns
        -------
        self : object
            calling this function sets the self.fitted_pipeline_ variable
            
        """
        # selected_features = features_dataset1
        # selected_target = target_dataset1
        
        # if not self._optimized_pipeline:
        #     raise RuntimeError(
        #         "There was an error in the AUTOQTL optimization process. Please make sure you passed the data to AUTOQTL correctly."
        #     )

        # else:
        #     self.fitted_pipeline_ = self._toolbox.compile(expr=self._optimized_pipeline) # changing the optimized DEAP pipeline  to sklearn pipeline

        #     with warnings.catch_warnings():
        #         warnings.simplefilter("ignore")
        #         self.fitted_pipeline_.fit(selected_features, selected_target) # the sklearn pipeline is fitted with the pipeline fit function

        #         if self.verbosity in [1, 3]:
        #             # Add an extra line of spacing if the progress bar was used
        #             if self.verbosity >=2:
        #                 print("")

        #             optimized_pipeline_str = self.clean_pipeline_string(
        #                 self._optimized_pipeline
        #             )
        #             #print("Best pipeline:", optimized_pipeline_str)
        #             #print("Score of pipeline on two datasets respectively: ", self._optimized_pipeline_score)

        #         # Store, fit and display the entire Pareto front 
        #         self.pareto_front_fitted_pipelines_ = {} # contains the fitted pipelines present in the pareto front
        #         pareto_front_pipeline_str = {} # contains the pareto front pipeline strings

        #         for pipeline in self._pareto_front.items:
        #             self.pareto_front_fitted_pipelines_[
        #                 str(pipeline)
        #             ] = self._toolbox.compile(expr=pipeline)

        #             with warnings.catch_warnings():
        #                 warnings.simplefilter("ignore")
        #                 self.pareto_front_fitted_pipelines_[str(pipeline)].fit(
        #                     selected_features, selected_target
        #                 )
        #                 pareto_front_pipeline_str[str(pipeline)] = self.clean_pipeline_string(pipeline)

                
        #                 #print("Pareto front individuals: ", pareto_front_pipeline_str)
        #                 # can print the fitness tuples of those pipelines
        #         # Printing the final pipeline
        #         print("Final Pareto Front at the end of the optimization process: ")
        #         for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
        #             pipeline_to_be_printed = self.print_pipeline(pipeline)
        #             print('\nTest R^2 = {0},\tDifference Score = {1},\tPipeline: {2}'.format(
        #                     pipeline_scores.wvalues[0],
        #                     pipeline_scores.wvalues[1],
        #                     pipeline_to_be_printed))

        ################ NEW VERSION OF THE METHOD ###################
        # all the Pareto front pipelines are fitted using dataset1, i.e, training dataset
        selected_features = features_dataset1 
        selected_target = target_dataset1
        
        if not self._optimized_pipeline:
            raise RuntimeError(
                "There was an error in the AUTOQTL optimization process. Please make sure you passed the data to AUTOQTL correctly."
            )

        else:
            self.fitted_pipeline_ = self._toolbox.compile(expr=self._optimized_pipeline) # changing the optimized DEAP pipeline  to sklearn pipeline

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.fitted_pipeline_.fit(selected_features, selected_target) # the sklearn pipeline is fitted with the pipeline fit function

                if self.verbosity in [1, 3]:
                    # Add an extra line of spacing if the progress bar was used
                    if self.verbosity >=2:
                        print("")

                    optimized_pipeline_str = self.clean_pipeline_string(
                        self._optimized_pipeline
                    )

                # Store, fit and display the entire Pareto front 
                self._pareto_front_fitted_pipelines_ = {} # contains the fitted pipelines present in the pareto front
                pareto_front_pipeline_str = {} # contains the pareto front pipeline strings

                for pipeline in self._pareto_front.items:
                    self._pareto_front_fitted_pipelines_[
                        str(pipeline)
                    ] = self._toolbox.compile(expr=pipeline)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self._pareto_front_fitted_pipelines_[str(pipeline)].fit(
                            selected_features, selected_target
                        )
                        pareto_front_pipeline_str[str(pipeline)] = self.clean_pipeline_string(pipeline)
    

                # store score 1 and score 2 from the final pareto front
                self.score1_final_list =[]
                self.score2_final_list = []

                # Printing the final pareto pipeline
                print("Final Pareto Front at the end of the optimization process: ")
                for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
                    #pipeline_to_be_printed = self.print_pipeline(pipeline) # change to EZ version
                    print('\nTest R^2 = {0},\tDifference Score = {1}'.format(
                            pipeline_scores.wvalues[0],
                            pipeline_scores.wvalues[1]))
                    
                    print(self.print_pipeline_hyper(pipeline))

                    # # print the names of the selected features if FS exists in the pipeline
                    # pipeline_sklearn = self._toolbox.compile(expr=pipeline)
                    # with warnings.catch_warnings():
                    #     warnings.simplefilter("ignore")
                    # pipeline_sklearn.fit(selected_features, selected_target)
                    # flag = False
                    # for name, transformer in pipeline_sklearn.steps:
                    #     if name=='variancethreshold' or name=='selectpercentile' or name=='featureencodingfrequencyselector':
                    #         selected_features_mask = transformer.get_support()
                    #         selected_features_names = features_dataset1.columns[selected_features_mask]
                    #         print("No of features selected in the pipeline: ", len(selected_features_names))
                    #         print("Feature names selected in the pipeline: ", selected_features_names.tolist())
                    #         flag = True

                    # if flag==False:
                    #     print("No feature selectors were selected in the pipeline")

                    self.score1_final_list.append(pipeline_scores.wvalues[0])
                    self.score2_final_list.append(pipeline_scores.wvalues[1])

    # function to print pipeline with hyperparameters
    def print_pipeline_hyper(self, individual, cleaned = False):
        
        """This function cleans up the pipeline representation.
        It also displays all the hyperparameters.
        
        """
        if not cleaned: # cleaned represents whether inputted individual has been compiled into scikit learn form yet
            dirty_string = self._toolbox.compile(expr=individual)
        else:
            dirty_string = individual

        savePrevLine = ""
        pretty_string = "" # initialize result variable
        count = 0
        for line in str(dirty_string).splitlines():
            if savePrevLine != "": # if previous line should be combined with current line
                line = savePrevLine + line.lstrip()
                savePrevLine = ""  

            if "'" in line: # if line includes extra operator to remove
                if ")" in line: # if extra operator and necessary operator are on same line
                    line = re.sub('\'[^>]+ ', '', line)
                else: #if extra operator comes on line before the necessary operator
                    line = re.sub('\'[^>]+,', '', line)
                    savePrevLine = line #saves current line to combine with next line
                    continue

            #remove score_func argument in selectpercentile pipelines
            if(re.search('SelectPercentile', line) is not None):
                if re.search('score_func', line) is None:
                    savePrevLine = line[:-1] + '),'
                    continue
                elif(re.search('percentile', line) is not None):
                    line = re.sub('score_func(.*)', '', line)
                else: #adds percentile=10 as default parameter in SelectPercentile
                    line = re.sub('score_func(.*)\>', 'percentile=10', line)
            
            #remove unnecessary open parentheses
            if(re.search('\([A-Z]', line) is not None):
                index = re.search('\([A-Z]', line).start()
                line = line[:index] + line[index + 1:]
            if(re.match('[a-z]', line.lstrip())):
                line = line[1:]
            
            #clean start of pipeline
            line = str.replace(line, "Pipeline(steps=[", "Pipeline steps: ") 

            #remove unnecessary end parentheses
            line = str.replace(line, "]", "")
            line = str.replace(line, "))", ")")
            line = str.replace(line, "))", ")")

            #add number labelling to each step and manage indents
            if(re.search("Pipeline steps: ", line) is not None):
                count = 1
                index = re.search("Pipeline steps: ", line).end()
                line = line[:index] + "\n     " + str(count) + ". " + line[index:]
            elif(re.search('   [A-Z]', line) is not None):
                count += 1
                line = "{:>8}{}".format(str(count) + ". ", line.lstrip())
            else: 
                line = line[8:]

            #combine with final result
            pretty_string = pretty_string + line + "\n"

        return pretty_string           
                        

    # To get a pipeline outputted in a desired format
    def print_pipeline(self, individual):
        """Print the pipeline in a user-friendly manner.
        
        Parameters
        ----------
        individual: Pareto front individuals
            Individual which should be represented by a pretty string
            
        Returns
        A string suitable for display
        
        """
        dirty_string = str(individual)

        parameter_prefixes = [
            (m.start(), m.end()) for m in re.finditer("[\w]+([\w])", dirty_string)
        ]
        substring = '__'
        pretty_string = ''
        for (start, end) in reversed(parameter_prefixes):
    
            if (substring in dirty_string[start:end]) or (dirty_string[start:end].isdigit()):
                continue
            elif (start, end) == parameter_prefixes[0]: 
                pretty_string = pretty_string + dirty_string[start:end] + "."
            else:
                pretty_string = pretty_string + dirty_string[start:end] + ' -> '
        return pretty_string


    # make the pipeline suitable for display
    def clean_pipeline_string(self, individual):
        """Provide a string of the individual without the parameter prefixes.
        Parameters
        ----------
        individual: individual
            Individual which should be represented by a pretty string
        Returns
        -------
        A string like str(individual), but with parameter prefixes removed.
        """
        dirty_string = str(individual)
        # There are many parameter prefixes in the pipeline strings, used solely for
        # making the terminal name unique, eg. LinearSVC__.
        
        parameter_prefixes = [
            (m.start(), m.end()) for m in re.finditer(", [\w]+__", dirty_string)
        ]
        # We handle them in reverse so we do not mess up indices
        pretty = dirty_string
        for (start, end) in reversed(parameter_prefixes):
            pretty = pretty[: start + 2] + pretty[end:]

        return pretty


    # scoring function - which returns score for each pipeline in the pareto front
    def score(self, holdout_features, holdout_target):
        """Returns the holdout R^2 values on the holdout dataset by using each pareto front pipeline trained using the training data.
        
        Parameters
        __________
        holdout_features: array-like {n_samples, n_features}
            Feature matrix of the holdout set
        holdout_target: array-like {n_samples}
            List of class labels for prediction in the holdout set

        Returns
        _______
        R^2 value: list
            List of the estimated holdout set R^2 values for each pareto front pipelines

        """
        print("Results of calling the score function: ")
        if len(self._pareto_front_fitted_pipelines_) == 0:
            raise RuntimeError(
                               "No pipeline has been optimized and fitted yet. Please call fit() first.")
        
        holdout_features, holdout_target = self._check_dataset(holdout_features, holdout_target, sample_weight = None)

        # If the scoring function is a string, we must adjust to use the sklearn
        # scoring interface
        if isinstance(self.scoring_function, str):
            #scorer = SCORERS[self.scoring_function]
            scorer = get_scorer(self.scoring_function)
        elif callable(self.scoring_function):
            scorer = self.scoring_function
        else:
            raise RuntimeError(
                "The scoring function should either be the name of a scikit-learn scorer or a scorer object"
            )
        
        holdout_score_list = []
        for fitted_pipeline in self._pareto_front_fitted_pipelines_.values():
            print(re.sub("Pipeline steps:", "", self.print_pipeline_hyper(fitted_pipeline, cleaned=True)))
            score = scorer(
            fitted_pipeline,
            holdout_features.astype(np.float64),
            holdout_target.astype(np.float64),
            )
            print("Holdout R\u00b2: ", score)
            holdout_score_list.append(score)

        return holdout_score_list
    
    # function to print the holdout score for the pipeline asked by the user
    def score_user_choice(self, holdout_features, holdout_target, pipeline_no):
        """Returns the holdout R^2 values on the holdout dataset by using the specified Pareto front pipeline trained using the training data.
        
        Parameters
        __________
        holdout_features: array-like {n_samples, n_features}
            Feature matrix of the holdout set
        holdout_target: array-like {n_samples}
            List of class labels for prediction in the holdout set
        pipeline_no: The Pareto front pipeline number to be used as the optimal pipeline to evaluat ethe holdout dataset.

        Returns
        _______
        R^2 value: list
            List of the estimated holdout set R^2 values for the specified Pareto front pipeline.
        
        """
        if len(self._pareto_front_fitted_pipelines_) == 0:
            raise RuntimeError(
                               "No pipeline has been optimized and fitted yet. Please call fit() first.")
        
        holdout_features, holdout_target = self._check_dataset(holdout_features, holdout_target, sample_weight = None)

        # If the scoring function is a string, we must adjust to use the sklearn
        # scoring interface
        if isinstance(self.scoring_function, str):
            #scorer = SCORERS[self.scoring_function]
            scorer = get_scorer(self.scoring_function)
        elif callable(self.scoring_function):
            scorer = self.scoring_function
        else:
            raise RuntimeError(
                "The scoring function should either be the name of a scikit-learn scorer or a scorer object"
            )
        pareto_front_fitted_pipelines_key_list = list(self._pareto_front_fitted_pipelines_.keys())
        score = scorer(
            self._pareto_front_fitted_pipelines_[pareto_front_fitted_pipelines_key_list[pipeline_no-1]],
            holdout_features.astype(np.float64),
            holdout_target.astype(np.float64),
            )
        
        return score
    
    # function definition for predict to get the predicted target values for the holdout dataset
    def predict(self, holdout_features):
        """Returns the predicted target values on the holdout dataset by using each pareto front pipeline trained using the training data.
        
        Parameters
        __________
        holdout_features: array-like {n_samples, n_features}
            Feature matrix of the holdout set

        Returns
        _______
        predicted_target: [[]]
            Dictionary of predicted target values for each pareto front pipeline.

        """
        if len(self._pareto_front_fitted_pipelines_) == 0:
            raise RuntimeError(
                               "No pipeline has been optimized and fitted yet. Please call fit() first.")
        
        holdout_features = self._check_dataset(holdout_features, target=None, sample_weight=None)

        predicted_target_dict = {} # The key of the dictionary will be the pipeline name and the value will be a list of the predicted target values.
        i = 0
        for fitted_pipeline in self._pareto_front_fitted_pipelines_.values():
            i = i + 1
            predicted_target_list_per_pipeline = fitted_pipeline.predict(holdout_features)
            #pipeline_name_key = self.print_pipeline_hyper(fitted_pipeline, cleaned=True)
            pipeline_name_key = "Pipeline " + "#" + str(i)
            predicted_target_dict[pipeline_name_key] = (np.round(predicted_target_list_per_pipeline, 6)).tolist()
            
           
        return predicted_target_dict
    
    # function to predict the target values based on a particular pareto front pipeline as inputed by the user
    def predict_user_choice(self, holdout_features, pipeline_no):
        """Returns the predicted target values on the holdout dataset by using the chosen Pareto front pipeline trained using the training data.
        
        Parameters
        __________
        holdout_features: array-like {n_samples, n_features}
            Feature matrix of the holdout set

        Returns
        _______
        predicted_target: [[]]
            Dictionary of predicted target values for the chosen Pareto front pipeline.

        """
        if len(self._pareto_front_fitted_pipelines_) == 0:
            raise RuntimeError(
                               "No pipeline has been optimized and fitted yet. Please call fit() first.")
        
        # holdout_features, holdout_target = self._check_dataset(holdout_features, holdout_target, sample_weight = None)
        holdout_features = self._check_dataset(holdout_features, target=None, sample_weight = None)
        pareto_front_fitted_pipelines_key_list = list(self._pareto_front_fitted_pipelines_.keys())
        predicted_target_list = self._pareto_front_fitted_pipelines_[pareto_front_fitted_pipelines_key_list[pipeline_no-1]].predict(holdout_features)
              
        return np.round(predicted_target_list, 6).tolist()
    
    # function to give the SHAP feature importance bar graph of a particular pipeline as inputted by the user
    def shap_feature_importance_user_choice(self, features, target, pipeline_no):
        """Gives the SHAP summary plot for the chosen pipeline.
        
        """
        if len(self._pareto_front_fitted_pipelines_) == 0:
            raise RuntimeError(
                               "No pipeline has been optimized and fitted yet. Please call fit() first.")
        features, target = self._check_dataset(features, target, sample_weight=None)
        pareto_front_fitted_pipelines_key_list = list(self._pareto_front_fitted_pipelines_.keys())
        selected_pipeline = self._pareto_front_fitted_pipelines_[pareto_front_fitted_pipelines_key_list[pipeline_no-1]]

        # compute SHAP
        num_features = features.shape[1]
        max_evals = max(500, 2 * num_features + 1)
        explainer = shap.Explainer(selected_pipeline.predict, features)
        shap_values = explainer(features, max_evals=max_evals)
        fig, ax = plt.subplots()
        shap.summary_plot(shap_values, features, plot_type='bar', show=False)
        plt.tight_layout()

        # Save the plot to a file
        #plt.savefig(f"Pipeline{pipeline_no}.png")
        #plt.close()
        return plt # added a line to the plot


    # This function initializes all the variables required to use the fit()
    def _fit_init(self):
        """Initialization for fit function. """

        if not self.warm_start or not hasattr(self, "_pareto_front"): # if we do not want to use the previous generation populations and a pareto front is not created already
            self._pop = [] # Population of individuals
            self._pareto_front = None # List of non-dominated individuals making up the Pareto front
            self._last_optimized_pareto_front = None # List of pipeline scores of the pipelines present in the pareto front. Basically list of list [[score_d1, score_d2], []]
            self._last_optimized_pareto_front_n_gens = 0 # Generation number of the last optimized pareto front

            self._setup_config(self.config_dict) # Set up the configuartion dictionary, containing all the selectors, transformers and ML methods

            self._setup_template(self.template)

            self.operators = [] # List of all the operator classes which will be used to construct the DEAP pipeline
            self.arguments = [] # List of argument classes. Basically a list of list. One list for each operator.

            make_pipeline_func = self._get_make_pipeline_func() # The function which will be used to generate the sklearn pipeline

            for key in sorted(self._config_dict.keys()):
                op_class, arg_types = AUTOQTLOperatorClassFactory(
                    key,
                    self._config_dict[key],
                    BaseClass=Operator,
                    ArgBaseClass=ARGType,
                    verbose=self.verbosity,
                ) # For each key value pair in the config dictionary we generate the corresponding Operator class and Argument classes

                if op_class:
                    self.operators.append(op_class)
                    self.arguments += arg_types
            
            self.operators_context = {
                "make_pipeline" : make_pipeline_func,
                "make_union" : make_union,
                "FunctionTransformer" : FunctionTransformer,
                "copy" : copy,
            } # Which function to use when these keys are encountered. 

            self._setup_pset() # Setup the primitive set to contain the operators and the arguments
            self._setup_toolbox() # Setup the toolbox to contain the tools such as mutation, crossover, etc

            self.evaluated_individuals_ = {} # Dictionary of individuals that have already been evaluated in previous generations or previous runs

        self._optimized_pipeline = None # the best pipeline among all the individuals in the pareto front
        self._optimized_pipeline_score = None # score of the optimized pipeline, two R2 values for two datasets
        self._exported_pipeline_text = [] # Saves the entire pipeline code in text format to output in a file
        self.fitted_pipeline_ = None # the fitted version of the optimized pipeline which is used in score and predict functions
        self._fitted_imputer = None # the kind of imputer to be used in case imputation is required
        self._imputed = False # to know if imputer was used or not
        self._memory = None # initial memory setting for sklearn pipeline

        self.output_best_pipeline_period_seconds = 30 # don't save periodic pipelines more often than this
        self._max_mut_loops = 50 # Try crossover and mutation at most this many times for any given individual (or pair of individuals)

        if self.max_time_mins is None and self.generations is None:
            raise ValueError(
                "Either the parameter generations should be set or a maximum evaluation time should be defined via max_time_mins"
            )
        
        # If no. of generations is not specified and run-time limit is specified, schedule AutoQTL to run till it automatically interrupts itself when the timer runs out
        if self.max_time_mins is not None and self.generations is None:
            self.generations = 1000000

        # Put in check for version check later on

        # check the sum of mutation and crossover rates
        if self.mutation_rate + self.crossover_rate > 1:
            raise ValueError(
                "The sum of the crossover and mutation probablities must be <=1.0. "
            )
        
        self._pbar = None # declare the progress bar

        # setting up the log file
        if not self.log_file:
            self.log_file_ = sys.stdout
        elif isinstance(self.log_file, str):
            self.log_file_ = open(self.log_file, "w")
        else:
            self.log_file_ = self.log_file

        self._setup_scoring_function(self.scoring) # setup scoring function

        # setup subsample value if inputted by the user
        if self.subsample <= 0.0 or self.subsample > 1.0:
            raise ValueError(
                "The subsample ratio of the training instance must be in the range (0.0, 1.0]. "
            ) 

        # Put in check for no.of jobs later when using dask

    
    # Function to perform a pretest on a sample of data to verify pipelines work with the passed data set
    def _init_pretest(self, features, target):
        """Set the sample of data used to verify whether pipelines work with the passed data set. We use one dataset in the pretest. 
        
        """
        #raise ValueError("Use AutoQTLRegressor")
        print("Use AutoQTLRegressor ")
        """self.pretest_X, _, self.pretest_y, _ = \
                train_test_split(
                                features,
                                target,
                                random_state=self.random_state,
                                test_size=None,
                                train_size=min(50,int(0.9*features.shape[0]))
                                )"""

    
    # Function to impute missing values
    def _impute_values(self, features):
        """Impute missing values in a feature set.
        
        Parameters
        ----------
        features : array_like {n_samples, n_features}
            A feature matrix
            
        Returns
        -------
        array-like {n_samples, n_features}
        
        """
        if self.verbosity > 1:
            print("Imputing missing values in feature set")

        if self._fitted_imputer is None:
            self._fitted_imputer = SimpleImputer(strategy="most_frequent")
            self._fitted_imputer.fit(features)

        return self._fitted_imputer.transform(features)

    # Function to check for validity of dataset
    def _check_dataset(self, features, target, sample_weight=None):
        """Check if a dataset has a valid feature set and labels.
        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        target: array-like {n_samples} or None
            List of class labels for prediction
        sample_weight: array-like {n_samples} (optional)
            List of weights indicating relative importance
        Returns
        -------
        (features, target)
        """
        # Check sample_weight
        if sample_weight is not None:
            try:
                sample_weight = np.array(sample_weight).astype("float")
            except ValueError as e:
                raise ValueError(
                    "sample_weight could not be converted to float array: %s" % e
                )
            if np.any(np.isnan(sample_weight)):
                raise ValueError("sample_weight contained NaN values.")
            try:
                check_consistent_length(sample_weight, target)
            except ValueError as e:
                raise ValueError(
                    "sample_weight dimensions did not match target: %s" % e
                )

        # check for features
        if isinstance(features, np.ndarray):
                if np.any(np.isnan(features)):
                    self._imputed = True
        elif isinstance(features, DataFrame): 
                if features.isnull().values.any():
                    self._imputed = True

        if self._imputed:
                features = self._impute_values(features)

        # check for target
        try:
            if target is not None:
                X, y = check_X_y(features, target, accept_sparse=True, dtype=None)
                if self._imputed:
                    return X, y
                else:
                    return features, target
            else:
                X = check_array(features, accept_sparse=True, dtype=None)
                if self._imputed:
                    return X
                else:
                    return features
        except (AssertionError, ValueError):
            raise ValueError(
                "Error: Input data is not in a valid format. Please confirm "
                "that the input data is scikit-learn compatible. For example, "
                "the features must be a 2-D array and target labels must be a "
                "1-D array."
            )

    
    # the fit function of AutoQTL
    def fit(self, features, target, random_state, test_size = 0.5, sample_weight = None):
        """Fit an optimized machine learning pipeline.
        
        """
        # split the features, target into training and testing according to the test_size given.
        self.features_dataset1, self.features_dataset2, self.target_dataset1, self.target_dataset2 = train_test_split(features, target, test_size=test_size, random_state=random_state)
        self._fit_init()
        features_dataset1, target_dataset1 = self._check_dataset(self.features_dataset1, self.target_dataset1, sample_weight)
        features_dataset2, target_dataset2 = self._check_dataset(self.features_dataset2, self.target_dataset2, sample_weight)
        
        self._init_pretest(features_dataset1, target_dataset1)

        # Randomly collect a subsample of training sample for pipeline optimization process. Do it for both the dataset
        if self.subsample < 1.0:
            features_dataset1, _, target_dataset1, _ = train_test_split(
                features_dataset1,
                target_dataset1,
                train_size=self.subsample,
                test_size=None,
                random_state=self.random_state,
            )

            features_dataset2, _, target_dataset2, _ = train_test_split(
                features_dataset2,
                target_dataset2,
                train_size=self.subsample,
                test_size=None,
                random_state=self.random_state,
            )

            # Raise a warning message if the training size is less than 1500 when subsample is not default value
            if features_dataset1.shape[0] < 1500 or features_dataset2.shape[0] < 1500:
                print(
                    "Warning: Although subsample can accelerate pipeline optimization, "
                    "too small training sample size may cause unpredictable effect on maximizing "
                    "score in pipeline optimization process. Increasing subsample ratio may get "
                    "a more reasonable outcome from optimization process in AUTOQTL. "
                )

        # set the seed for the GP run
        if self.random_state is not None:
            random.seed(self.random_state) # deap uses random
            np.random.seed(self.random_state)

        self._start_datetime = datetime.now() # the datetime at the beginning of the optimization process
        self._last_pipeline_write = self._start_datetime # the last time a pipeline was recorded

        # register the "evaluate" operator in the toolbox
        self._toolbox.register(
            "evaluate",
            self._evaluate_individuals,
            features_dataset1=features_dataset1,
            target_dataset1=target_dataset1,
            features_dataset2=features_dataset2,
            target_dataset2=target_dataset2,
            sample_weight=sample_weight,
        )

        # assign population, self._pop can only be not None if warm_start is enabled
        if not self._pop:
            self._pop = self._toolbox.population(n=self.population_size)

        def pareto_eq(ind1, ind2):
            """Determine whether two individuals are equal on the Pareto front.
            
            Parameters
            ----------
            ind1 : DEAP individual from the GP population
                First individual to compare
            ind2 : DEAP individual from the GP population
                Second individual to compare
            
            Returns
            -------
            individuals_equal : bool
                Boolean indicating whether the two individuals are equal on the Pareto front
                
            """
            return np.allclose(ind1.fitness.values, ind2.fitness.values) # checks whether the fitness values of two individuals are equal or not

        # Generate new pareto front if it doesn't alreday exist for warm start
        if not self.warm_start or not self._pareto_front:
            self._pareto_front = tools.ParetoFront(similar=pareto_eq)

        # Set lambda_ (offspring size in GP) equal to pupulation_size by default
        if not self.offspring_size:
            self._lambda = self.population_size
        else:
            self._lambda = self.offspring_size

        # Start the progress bar
        if self.max_time_mins:
            total_evals = self.population_size
        else:
            total_evals = self._lambda * self.generations + self.population_size

        self._pbar = tqdm(
            total=total_evals,
            unit="pipeline",
            leave=False,
            file=self.log_file_,
            disable=not(self.verbosity >=2),
            desc="Optimization Progress",
        )

        try:
            with warnings.catch_warnings():
                self._setup_memory()
                warnings.simplefilter("ignore")
                self._pop, _ = eaMuPlusLambda(
                    population=self._pop,
                    toolbox=self._toolbox,
                    mu=self.population_size,
                    lambda_=self._lambda,
                    cxpb=self.crossover_rate,
                    mutpb=self.mutation_rate,
                    ngen=self.generations,
                    pbar=self._pbar,
                    halloffame=self._pareto_front,
                    verbose=self.verbosity,
                    per_generation_function=self._check_periodic_pipeline,
                    log_file=self.log_file_,
                )
        # Allow for certain exceptions to signal a premature fit() cancellation
        except(KeyboardInterrupt, SystemExit, StopIteration) as e:
            if self.verbosity > 0:
                self._pbar.write("", file=self.log_file_)
                self._pbar.write(
                    "{}\nAutoQTL closed prematurely. Will use the current best pipleine.".format(
                    e),
                    file=self.log_file_,
                )
        finally:
            # Clean population for the next call if warm_start=False
            if not self.warm_start:
                self._pop = []

            # keep trying 10 times in case weird things happened like multiple CTRL+C or exceptions
            attempts = 10
            for attempt in range(attempts):
                try:
                    # Close the progress bar
                    # Standard truthiness checks won't work for tqdm
                    if not isinstance(self._pbar, type(None)):
                        self._pbar.close()

                    self._update_top_pipeline()
                    self._summary_of_best_pipeline(features_dataset1, target_dataset1, features_dataset2, target_dataset2)
                    self._cleanup_memory()
                    break

                except (KeyboardInterrupt, SystemExit, Exception) as e:
                    # raise the exception if it's our last attempt
                    if attempt == (attempts - 1):
                        raise e
            return self

    def _setup_memory(self):
        """Setup Memory object for memory caching.
        """
        if self.memory:
            if isinstance(self.memory, str):
                if self.memory == "auto":
                    # Create a temporary folder to store the transformers of the pipeline
                    self._cachedir = mkdtemp()
                else:
                    if not os.path.isdir(self.memory):
                        try:
                            os.makedirs(self.memory)
                        except:
                            raise ValueError(
                                "Could not create directory for memory caching: {}".format(
                                    self.memory
                                )
                            )
                    self._cachedir = self.memory

                self._memory = Memory(location=self._cachedir, verbose=0)
            elif isinstance(self.memory, Memory):
                self._memory = self.memory
            else:
                raise ValueError(
                    "Could not recognize Memory object for pipeline caching. "
                    "Please provide an instance of joblib.Memory,"
                    ' a path to a directory on your system, or "auto".'
                )


    def _cleanup_memory(self):
        """Clean up caching directory at the end of optimization process only when memory='auto'"""
        if self.memory == "auto":
            rmtree(self._cachedir)
            self._memory = None

    # # Trying out feature importance with permutation importance score
    # def get_feature_importance(self, X, y, random_state):
    #     """
    #      Parameters
    #     ----------
    #     """
        
    #     self.pipeline_for_feature_importance_ = {}
    #     self.fitted_pipeline_for_feature_importance =[]
    #     """for key, value in self.pareto_front_fitted_pipelines_.items():
    #         pipeline_estimator = value
        
    #     print(pipeline_estimator)"""
        
    #     for pipeline in self._pareto_front.items:
    #                 self.pipeline_for_feature_importance_[
    #                     str(pipeline)
    #                 ] = self._toolbox.compile(expr=pipeline)

    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore")
    #                     self.pipeline_for_feature_importance_[str(pipeline)].fit(
    #                         X, y
    #                     )
    #                     self.fitted_pipeline_for_feature_importance.append(self.pipeline_for_feature_importance_[str(pipeline)].fit(
    #                         X, y
    #                     ))
        
    #     #print(self.pipeline_for_feature_importance_) 
    #     #print(self.fitted_pipeline_for_feature_importance[0])

    #     """for key, value in self.pipeline_for_feature_importance_.items():
    #         pipeline_estimator = value"""
                       
    #     pipeline_estimator = self.fitted_pipeline_for_feature_importance[2]
    #     print(pipeline_estimator)


    #      # Printing the final pipeline
    #     print("Final Pareto Front at the end of the optimization process: ")
    #     for pipeline, pipeline_scores in zip(self._pareto_front.items, reversed(self._pareto_front.keys)):
    #         pipeline_to_be_printed = self.print_pipeline(pipeline)
    #         print('\nScore on D1 = {0},\tScore on D2 = {1},\tPipeline: {2}'.format(
    #                         pipeline_scores.wvalues[0],
    #                         pipeline_scores.wvalues[1],
    #                         pipeline_to_be_printed))
    #     # Testing 
    #     """estimators = [('feature_extraction', VarianceThreshold(threshold=0.25)), ('regression', LinearRegression())]
    #     pipeline = Pipeline(estimators)
    #     pipeline.fit(X, y)"""
    #     # Putting output to a text file
    #     """file_path = 'output.txt'
    #     sys.stdout = open(file_path, "w")
    #     for fitted_pipeline in self.fitted_pipeline_for_feature_importance:
    #         print("The Pipeline being evaluated: ", fitted_pipeline)
    #         permutation_importance_object = permutation_importance(estimator=fitted_pipeline, X=X, y=y, n_repeats=5, random_state=random_state)
    #         for i in permutation_importance_object.importances_mean.argsort()[::-1]:
    #             print(f"{X.columns[i]:<8}"
    #                 f"{permutation_importance_object.importances_mean[i]:.3f}")"""

    #     permutation_importance_object = permutation_importance(estimator=pipeline_estimator, X=X, y=y, n_repeats=5, random_state=random_state)
    #     for i in permutation_importance_object.importances_mean.argsort()[::-1]:
    #             print(f"{X.columns[i]:<8}"
    #                 f"{permutation_importance_object.importances_mean[i]:.3f}")
        
        
    
    # def get_shap_values(self, X, y):
    #     estimators = [('feature_extraction', VarianceThreshold(threshold=0.25)), ('regression', LinearRegression())]
    #     pipeline = Pipeline(estimators)

    #     pipeline.fit(X, y)

    #     print(X.shape[1])
    #     X_background = shap.utils.sample(X, 2500)
    #     num_features = X.shape[1]
    #     max_evals = max(500, 2 * num_features + 1)
    #     explainer = shap.Explainer(self.fitted_pipeline_.predict, X)
    #     shap_values = explainer(X, max_evals=max_evals)
    #     shap.summary_plot(shap_values, X, plot_type='bar')

    
     #############################################################################################################################
    # # Getting test R^2 values (basically holdout R^2 values) for the pipelines in the pareto front
    # def get_test_r2(self, d1_X, d1_y, d2_X, d2_y, holdout_X, holdout_y, feature_80, target_80, entire_X, entire_y):
        
    #     self.final_pareto_pipelines_testR2 = {}
    #     self.fitted_final_pareto_pipelines_testR2 =[]
        
    #     for pipeline in self._pareto_front.items:
    #                 self.final_pareto_pipelines_testR2[
    #                     str(pipeline)
    #                 ] = self._toolbox.compile(expr=pipeline)

    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore")
    #                     self.final_pareto_pipelines_testR2[str(pipeline)].fit(
    #                         feature_80, target_80
    #                     )
    #                     self.fitted_final_pareto_pipelines_testR2.append(self.final_pareto_pipelines_testR2[str(pipeline)].fit(
    #                         feature_80, target_80
    #                     ))

    #     final_output_file_path = 'EvaluationOnHoldout.txt'
    #     sys.stdout = open(final_output_file_path, "w")

        

    #     for pareto_pipeline in self.fitted_final_pareto_pipelines_testR2:
    #         print("\n The Pipeline being evaluated: \n", pareto_pipeline)
    #         #score = partial_wrapped_score(pareto_pipeline, holdout_X, holdout_y)
    #         score_80 = pareto_pipeline.score(feature_80, target_80)
    #         score_holdout_entireDataTrained = pareto_pipeline.score(holdout_X, holdout_y)
            
    #         print("\n Entire dataset(80%) R^2 trained on entire dataset(80%): ", score_80)
    #         print("\n Holdout data R^2 trained on entire dataset(80%): ", score_holdout_entireDataTrained)

    #         # To get the D1 train score
    #         pareto_pipeline.fit(d1_X, d1_y)
    #         score_d1_trained_d1 = pareto_pipeline.score(d1_X, d1_y)
    #         print("\n Dataset D1 score on trained D1: ", score_d1_trained_d1)


    #         # To get entire datatset
    #         pareto_pipeline.fit(entire_X, entire_y)
    #         score_full_trained_full = pareto_pipeline.score(entire_X, entire_y)
    #         print("\n Entire dataset R^2 using pipeline: ", score_full_trained_full)

    # def get_permutation_importance(self, X, y, random_state):
    #     self.pipeline_for_feature_importance_ = {}
    #     self.fitted_pipeline_for_feature_importance =[]
    #     for pipeline in self._pareto_front.items:
    #                 self.pipeline_for_feature_importance_[
    #                     str(pipeline)
    #                 ] = self._toolbox.compile(expr=pipeline)

    #                 with warnings.catch_warnings():
    #                     warnings.simplefilter("ignore")
    #                     self.pipeline_for_feature_importance_[str(pipeline)].fit(
    #                         X, y
    #                     )
    #                     self.fitted_pipeline_for_feature_importance.append(self.pipeline_for_feature_importance_[str(pipeline)].fit(
    #                         X, y
    #                     ))
    #     file_path = 'PermutationFeatureImportance.txt'
    #     sys.stdout = open(file_path, "w")

    #     # Permutation Feature Importance
    #     print("Feature Importance: \n ")
    #     for fitted_pipeline in self.fitted_pipeline_for_feature_importance:
    #         print("\nThe Pipeline being evaluated: \n", fitted_pipeline)
    #         permutation_importance_object = permutation_importance(estimator=fitted_pipeline, X=X, y=y, n_repeats=5, random_state=random_state)
    #         for i in permutation_importance_object.importances_mean.argsort()[::-1]:
    #             print(f"{X.columns[i]:<20}"
    #                 f"{permutation_importance_object.importances_mean[i]:.3f}")

    # Getting shap values in a separate text file and shap plots in a folder
    def get_shap_values(self, X, y, random_state):
        self.pipeline_for_feature_importance_ = {}
        self.fitted_pipeline_for_feature_importance =[]
        for pipeline in self._pareto_front.items:
                    #print(pipeline)
                    self.pipeline_for_feature_importance_[
                        str(pipeline)
                    ] = self._toolbox.compile(expr=pipeline)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.pipeline_for_feature_importance_[str(pipeline)].fit(
                            X, y
                        )
                        self.fitted_pipeline_for_feature_importance.append(self.pipeline_for_feature_importance_[str(pipeline)].fit(
                            X, y
                        ))
        file_path = 'ShapFeatureImportance.txt'
        sys.stdout = open(file_path, "w")

        # Shapley values
        print("Shapley Values")
        num_features = X.shape[1]
        max_evals = max(500, 2 * num_features + 1)
        save_folder = "ShapDiagrams"
        if not os.path.exists(save_folder):
            os.makedirs(save_folder)

        # try clearing the figure so that the pareto plot doesn't get superimposed
        plt.clf()
        
        i = 1 # To concatenate pipeline numbers in the png file names
        #fitted_pipeline = self.fitted_pipeline_for_feature_importance[6]
        for fitted_pipeline in self.fitted_pipeline_for_feature_importance:
            print("\nThe pipeline being evaluated: \n", fitted_pipeline)
            explainer = shap.Explainer(fitted_pipeline.predict, X)
            shap_values = explainer(X, max_evals=max_evals)
            # To get the values before making the summary plot
            vals = np.abs(shap_values.values).mean(0)
            shap_feature_importance = pd.DataFrame(list(zip(X.columns,vals)), columns=['col_name', 'feature_importance_vals'])
            shap_feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
            print(shap_feature_importance)
            # Saving the summary plot 
            shap.summary_plot(shap_values, X, plot_type='bar', show=False)
            plt.tight_layout()
            plt.savefig(f"{save_folder}/Pipeline{i}.png")
            i = i+ 1

    # Function to make graph based on average shap values for each feature considering all the pipelines in the run
    def average_feature_importance(self, X, y):
        self.pipeline_for_feature_importance_ = {}
        self.fitted_pipeline_for_feature_importance =[]
        for pipeline in self._pareto_front.items:
                    self.pipeline_for_feature_importance_[
                        str(pipeline)
                    ] = self._toolbox.compile(expr=pipeline)

                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        self.pipeline_for_feature_importance_[str(pipeline)].fit(
                            X, y
                        )
                        self.fitted_pipeline_for_feature_importance.append(self.pipeline_for_feature_importance_[str(pipeline)].fit(
                            X, y
                        ))

        # starting with the shap calculations
        num_features = X.shape[1]
        max_evals = max(500, 2*num_features + 1)

        # dictionary to store column name (feature) as key and list of shap values as value
        feature_name_importance_dict = {feature_name : [] for feature_name in X.columns}    

        # calculating the shap values for each pipeline
        for fitted_pipeline in self.fitted_pipeline_for_feature_importance:
            explainer = shap.Explainer(fitted_pipeline.predict, X)
            shap_values = explainer(X, max_evals=max_evals)
            vals = np.abs(shap_values.values).mean(0)
            for i in range(0, len(vals)):
                feature_name_importance_dict[X.columns[i]].append(round(vals[i], 6))
            #print(feature_name_importance_dict)

        # For making the graph using the dictionary
        x = [] # list of values for x axis
        y = [] # list of values for y axis

        # filling the lists, x will have the feature names and y will have the importance values
        for key in feature_name_importance_dict.keys():
            x.append(key)
            y.append(round(statistics.mean(feature_name_importance_dict[key]), 4))
        
        # clear the existing plot so that there is no overlapping
        plt.clf()

        plt.figure(figsize=(10,6))
        # bar plot for feature importance in descending order
        df = pd.DataFrame({"Feature":x, "Importance Score":y})
        ax = sns.barplot(x='Importance Score', y='Feature', data=df, order=df.sort_values('Importance Score', ascending=False).Feature, color='pastel', palette="mako")
        ax.bar_label(ax.containers[0])
        plt.xlabel("Average shap importance score across pareto pipelines", size=15)
        plt.ylabel("Features", size=15)
        plt.title("Feature Importance Graph", size=18)
        plt.tight_layout()
        #plt.savefig("AvgShapGraph.png", dpi=100)
        return plt

    # function to convert a text file to a pdf file
    def convert_text_to_pdf(self, text_file_path, pdf_file_path, plot1_path,plot2_path):

        # # checking if the plot files are fine or not
        # plot1.savefig("ParetoPlot_check.png")
        # plot2.savefig("AvgShapPlot_check.png")

        # read the text file
        with open(text_file_path, 'r') as f:
            text = f.read()
        # create a PDF object
        pdf = FPDF()
        # add a new page
        pdf.add_page()
        # set the font and font size
        pdf.set_font('Arial', size=12)
        # write the text to the PDF
        pdf.write(5, text)

        # to add first plot to the PDF
        pdf.add_page()
        # plot1.savefig('plot1.png')
        # pdf.image('plot1.png', x=10, y=40, w=190) # just directly add the plot as a parameter, so change plot1.png to plot1
        pdf.image(plot1_path, x=10, y=40, w=190)


        # save and close the first plot
        # plot1.savefig('plot1.png')
        # plt.close(plot1)

        # to add second plot to the PDF
        pdf.add_page()
        # plot2.savefig('plot2.png')
        # pdf.image('plot2.png', x=10, y=40, w=190)
        # #plt.close(plot2)
        pdf.image(plot2_path, x=10, y=40, w=190)

        # save the PDF file
        pdf.output(pdf_file_path, 'F')

    
    # def get_final_output(self, d1_X, d1_y, d2_X, d2_y, holdout_X, holdout_y, d80_X, d80_y, entire_X, entire_y, filename = None):
    def get_final_output(self, entire_X, entire_y, holdout_X, holdout_y, file_path = None):

        # final_output_file_path = 'final_output.txt'
        # sys.stdout = open(final_output_file_path, "w")

        if file_path is None:
            final_output_file_path = ''
        else:
            final_output_file_path = file_path 

        text_output_path = final_output_file_path + 'final_output.txt'
        pdf_output_path = final_output_file_path + 'final_output.pdf'

        sys.stdout = open(text_output_path, "w")

        print("-------------     autoQTL output     -------------\n")

        # #file name
        # if filename is not None:
        #     print("File name:", filename, "\n")

        # Printing the size of the data
        print("Total number of samples in the dataset: ", entire_X.shape[0])
        print("Total number of features in the dataset: ", entire_X.shape[1])

        print("The entire dataset is split into 80-20. The 80% is used in the evolution and the 20% is used as a hold out to assess model performances.")
        print("The 80% data is further splitted equally into training and testing datasets.")
        print("Training data: 1st split of 80% data.")
        print("Testing data: 2nd split of 80% data.")
        print("Holdout data: 20% spilt of entire data.")

        #autoQTL parameters
        print("autoQTL parameters:")
        print("\tpopulation size =", self.population_size)
        print("\toffspring size =", self.offspring_size)
        print("\tgenerations =", self.generations)
        print("\tmutation rate =", self.mutation_rate)
        print("\tcrossover rate =", self.crossover_rate)
        print("\trandom state = ", self.random_state)


        #evolution history
        print("\n*************************************************")
        print("Evolution History:")
        format_row = "{:>10}{:>16}{:>18}"
        print(format_row.format("", "Best Test R\u00b2 score", "Best difference score"))
        format_row = "{:>10}{:>16.5f}{:>18.5f}"
        try:
            for log in self._logbook:
                if log.get('gen') == 0:
                    continue
                print(format_row.format(("Gen " + str(log.get('gen'))), log.get('topscore1'), log.get('topscore2')))
                return_logbook(clear = True) #clears logbook when this is run
        except AttributeError:
            logbook = return_logbook(clear = True) #clears logbook when this is run
            for log in logbook:
                if log.get('gen') == 0:
                    continue
                print(format_row.format(("Gen " + str(log.get('gen'))), log.get('topscore1'), log.get('topscore2')))

        #multiple linear regression statistics for comparison
        print("\n*************************************************")
        print("Multiple Linear Regression:")
        lrmod = LinearRegression()
        format_row = "     {:<50}{:>8.5f}"

        # # Different LR values
        # lrmod.fit(d1_X, d1_y)
        # print(format_row.format("Train R\u00b2: ", lrmod.score(d1_X, d1_y)))
        # print(format_row.format("Test R\u00b2: ", lrmod.score(d2_X,d2_y)))
        # print(format_row.format("Holdout R\u00b2: ", lrmod.score(holdout_X, holdout_y)))
        # # LR model trained and tested on entire dataset
        # lrmod.fit(entire_X, entire_y)
        # print(format_row.format("Entire Dataset R\u00b2: ", lrmod.score(entire_X, entire_y)))

        # Different LR values - using the already split datasets
        lrmod.fit(self.features_dataset1, self.target_dataset1)
        print(format_row.format("Train R\u00b2: ", lrmod.score(self.features_dataset1, self.target_dataset1)))
        print(format_row.format("Test R\u00b2: ", lrmod.score(self.features_dataset2, self.target_dataset2)))
        print(format_row.format("Holdout R\u00b2: ", lrmod.score(holdout_X, holdout_y)))
        # LR model trained and tested on entire dataset
        lrmod.fit(entire_X, entire_y)
        print(format_row.format("Entire Dataset R\u00b2: ", lrmod.score(entire_X, entire_y)))

        #final pareto front statistics
        print("\n*************************************************")
        print("Final Pareto Front Statistics:\n")
        
        #find regression and encoder types on final pareto front
        reg_list = []
        enc_list = []
        for pipeline, score1, score2 in zip(self._pareto_front.items, self.score1_final_list, self.score2_final_list):
            enc = 'a'
            reg = ""
            string = str(self.print_pipeline_hyper(pipeline))
            if re.search('LinearRegression', string) is not None:
                reg = 'LR'
            elif re.search('RandomForest', string) is not None:
                reg = 'RF'
            elif re.search('DecisionTree', string) is not None:
                reg = 'DT'
            
            if re.search('Encoder', string) is not None:
                enc = '3'
            if re.search('Heterosis|Recessive|Dominant', string) is not None:
                enc = '2'
            
            reg_list.append(reg)
            enc_list.append(enc)

        #print final pareto front statistics
        tot = len(self._pareto_front.items)
        print(tot, "pipelines on the final Pareto front\n")

        format_row = "{:>16}{:>8}{:>8}"
        print(format_row.format("", "Number", "Percent"))
        print("--------------------------------")
        
        format_row = "{:>16}{:>8}{:>8.0%}"
        print(format_row.format("LinearRegression", sum(x == 'LR' for x in reg_list), sum(x == 'LR' for x in reg_list)/tot))
        print(format_row.format("MachineLearning", sum(x != 'LR' for x in reg_list), sum(x != 'LR' for x in reg_list)/tot))
        print("--------------------------------")
        print(format_row.format("RandomForest", sum(x == 'RF' for x in reg_list), sum(x == 'RF' for x in reg_list)/tot))
        print(format_row.format("DecisionTree", sum(x == 'DT' for x in reg_list), sum(x == 'DT' for x in reg_list)/tot))
        print("--------------------------------")
        print(format_row.format("AdditiveEncoder", sum(x == 'a' for x in enc_list), sum(x == 'a' for x in enc_list)/tot))
        print(format_row.format("3-LevelEncoder", sum(x == '3' for x in enc_list), sum(x == '3' for x in enc_list)/tot))
        print(format_row.format("2-LevelEncoder", sum(x == '2' for x in enc_list), sum(x == '2' for x in enc_list)/tot))

        print("\nRange of Test R\u00b2:         ({:.5f}, {:.5f})".format(min(self.score1_final_list), max(self.score1_final_list)))
        print("Range of Difference Score: ({:.5f}, {:.5f})".format(min(self.score2_final_list), max(self.score2_final_list)))

        #pipelines on final pareto front
        print("\n*************************************************")
        print("Final Pareto Front:\n")
        i = 0
        for pipeline, score1, score2 in zip(self._pareto_front_fitted_pipelines_.values(), self.score1_final_list, self.score2_final_list):
            i = i +  1
            print("Pipeline #" + str(i) + ": ")
            print("Test R\u00b2:", score1, "|\tDifference Score:", score2)
            print(re.sub("Pipeline steps:", "", self.print_pipeline_hyper(pipeline, cleaned=True)))
            
            #usable_pipeline = self._toolbox.compile(expr=unclean_pipeline) # scikit learn pipeline
            usable_pipeline = pipeline
            fitted_pipeline = pipeline

            # print the names of the selected features if FS exists in the pipeline
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     fitted_pipeline = usable_pipeline.fit(d1_X, d1_y) # usuable pipeline fitted on the training dataset (D1)
            flag = False
            current_data_feature_names = entire_X.columns
            j = 0
            for name, transformer in usable_pipeline.steps:
                if name=='variancethreshold' or "variancethreshold" in name or name=='selectpercentile' or "selectpercentile" in name  or name=='featureencodingfrequencyselector' or "featureencodingfrequencyselector" in name:
                    j = j+1
                    selected_features_mask = transformer.get_support()
                    selected_features_names = current_data_feature_names[selected_features_mask]
                    if len(selected_features_names) == len(current_data_feature_names):
                        print("\tFeature Selector " +name, "threshold at level " +str(j), "didn't select any features")
                    else:
                        print("\tNumber of features selected by " +name, "after " +str(j), "level of feature selection in the pipeline: ", len(selected_features_names))
                        print("\tFeature names of the selected features by " +name, "after " +str(j), "level of feature selection in the pipeline: ", selected_features_names.tolist())
                    current_data_feature_names = selected_features_names
                    flag = True

            if flag==False:
                print("\tNo feature selectors in the pipeline")

            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     fitted_pipeline = usable_pipeline.fit(d80_X, d80_y)
                
            # score_80 = fitted_pipeline.score(d80_X, d80_y)
            # score_holdout_entireDataTrained = fitted_pipeline.score(holdout_X, holdout_y)

            # score_on_holdout = fitted_pipeline.score(holdout_X, holdout_y)
            # score_on_train = fitted_pipeline.score(d1_X, d1_y)
            score_on_train = fitted_pipeline.score(self.features_dataset1, self.target_dataset1)

            # print("\tHoldout R\u00b2: ", score_on_holdout)
            print("\tTrain R\u00b2: ", score_on_train)
            
            # print("\tEntire dataset(80%) R^2 trained on entire dataset(80%): ", score_80)
            # print("\tHoldout data R^2 trained on entire dataset(80%): ", score_holdout_entireDataTrained)

            # # To get the entire data R2
            # with warnings.catch_warnings():
            #     warnings.simplefilter("ignore")
            #     fitted_pipeline = usable_pipeline.fit(entire_X, entire_y)
            # score_on_entire_dataset = fitted_pipeline.score(entire_X, entire_y)
            # print("\tEntire dataset R\u00b2: ", score_on_entire_dataset)
            print("\n-------------------------------------------------")

        # call the plot_final_pareto() function to get the pareto plot to add to the pdf
        plot1 = self.plot_final_pareto() # final pareto front plot
        plot1_path = "plot1.png"
        plot1.savefig(plot1_path)
        plot1.close()

        #pareto_plot.savefig("ParetoPlot_check.png")
        plot2 = self.average_feature_importance(entire_X, entire_y) # avg SHAP importance plot
        plot2_path = "plot2.png"
        plot2.savefig(plot2_path)   
        plot2.close()  
        #avg_shap_plot.savefig("AvgShapPlot_check.png")

        # # define the path to the plots and save the plot images to separate file paths
        # plot1_path = "plot1.png"
        # plot2_path = "plot2.png"
        # plot1.savefig(plot1_path)
        # plot1.close()
        # plot2.savefig(plot2_path)   
        # plot2.close()    

        
        # close the file
        sys.stdout.close()
        # close the file and redirect standard output back to console
        sys.stdout.close()
        sys.stdout = sys.__stdout__

        # call the function to convert the text file made to a pdf file
        self.convert_text_to_pdf(text_output_path, pdf_output_path, plot1_path, plot2_path)

    #outputs a pareto front
    def plot_final_pareto(self):
        rangex = max(self.score1_final_list) - min(self.score1_final_list)
        rangey = max(self.score2_final_list) - min(self.score2_final_list)
        n = range(1, 1 + len(self.score1_final_list))
        plt.scatter(self.score1_final_list, self.score2_final_list, edgecolors='black')
        plt.plot(self.score1_final_list, self.score2_final_list, color='black', linestyle='dashed')  # Change color and linestyle here
        for i, txt in enumerate(n):
            plt.annotate(txt, (self.score1_final_list[i], self.score2_final_list[i]), 
                               xytext = (self.score1_final_list[i] - 0.05*rangex, self.score2_final_list[i] - 0.05*rangey)) # Adjust the xytext values here
        plt.title('autoQTL: Final Pareto Front')
        plt.xlabel('Test R2')
        plt.ylabel('Difference Score')
        # Set xlim and ylim in reverse order
        plt.xlim(min(self.score1_final_list) - 0.1*rangex, max(self.score1_final_list) + 0.1*rangex)
        plt.ylim(min(self.score2_final_list) - 0.1*rangey, max(self.score2_final_list) + 0.1*rangey)
        #plt.savefig('pareto_plot_BMIPathways_12.png')
        #plt.close()
        return plt
        
