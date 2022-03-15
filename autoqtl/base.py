"""This file is part of AUTOQTL library"""
import imp
import inspect
import os
import random
from subprocess import call
import numpy as np
import deap
from deap import base, creator, tools, gp
from sklearn import tree

from sklearn.base import BaseEstimator
from sklearn.metrics import SCORERS

from .gp_types import Output_Array
from .builtins.combine_dfs import CombineDFs

from .config.regressor import regressor_config_dict

"""Building up the initial GP. """

class AUTOQTLBase(BaseEstimator):
    """Automatically creates and optimizes machine learning pipelines using Genetic Programming. """
    
    regression = None # set to True by child classes. Will be set to false in case of classification, when included. (variable name classification in case of TPOT)

    def __init__(
        self,
        generations = 100,
        population_size = 100,
        offspring_size = None,
        mutation_rate = 0.9,
        crossover_rate = 0.1,
        scoring = None,
        #cv = 5
        subsample = 1.0,
        n_jobs = 1,
        max_time_mins = None,
        max_eval_time_mins = 5,
        random_state = None,
        config_dict = None,
        template = None,
        warm_start = False,
        memory = None,
        #use_dask = False
        periodic_checkpoint_folder = None,
        early_stop = None,
        verbosity = 0,
        disable_update_check = False,
        log_file = None,
    ):
        """Set up the genetic programming algorithm for pipeline optimization. All the parameters are initialized with the default values. 
        
        Parameters
        ----------
        generations: int or None, optional (default: 100)
            Number of iterations to the run pipeline optimization process.
            It must be a positive number or None. If None, the parameter
            max_time_mins must be defined as the runtime limit.
            Generally, AUTOQTL will work better when you give it more generations (and
            therefore time) to optimize the pipeline. AUTOQTL will evaluate
            POPULATION_SIZE + GENERATIONS x OFFSPRING_SIZE pipelines in total.
        
        population_size: int, optional (default: 100)
            Number of individuals to retain in the GP population every generation.
            Generally, AUTOQTL will work better when you give it more individuals
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
            problem. By default, accuracy is used for classification problems and
            mean squared error (MSE) for regression problems.

            Offers the same options as sklearn.model_selection.cross_val_score as well as
            a built-in score 'balanced_accuracy'. Classification metrics:

            ['accuracy', 'adjusted_rand_score', 'average_precision', 'balanced_accuracy',
            'f1', 'f1_macro', 'f1_micro', 'f1_samples', 'f1_weighted',
            'precision', 'precision_macro', 'precision_micro', 'precision_samples',
            'precision_weighted', 'recall', 'recall_macro', 'recall_micro',
            'recall_samples', 'recall_weighted', 'roc_auc']

            Regression metrics:

            ['neg_median_absolute_error', 'neg_mean_absolute_error',
            'neg_mean_squared_error', 'r2']
        
        cv: int or cross-validation generator, optional (default: 5)
            If CV is a number, then it is the number of folds to evaluate each
            pipeline over in k-fold cross-validation during the AUTOQTL optimization
             process. If it is an object then it is an object to be used as a
             cross-validation generator. (NOT USED)
        
        subsample: float, optional (default: 1.0)
            Subsample ratio of the training instance. Setting it to 0.5 means that AUTOQTL
            randomly collects half of training samples for pipeline optimization process.
        
        n_jobs: int, optional (default: 1)
            Number of CPUs for evaluating pipelines in parallel during the TPOT
            optimization process. Assigning this to -1 will use as many cores as available
            on the computer. For n_jobs below -1, (n_cpus + 1 + n_jobs) are used.
            Thus for n_jobs = -2, all CPUs but one are used.
        
        max_time_mins: int, optional (default: None)
            How many minutes TPOT has to optimize the pipeline.
            If not None, this setting will allow AUTOQTL to run until max_time_mins minutes
            elapsed and then stop. AUTOQTL will stop earlier if generationsis set and all
            generations are already evaluated.
        
        max_eval_time_mins: float, optional (default: 5)
            How many minutes AUTOQTL has to optimize a single pipeline.
            Setting this parameter to higher values will allow AUTOQTL to explore more
            complex pipelines, but will also allow TPOT to run longer.
        
        random_state: int, optional (default: None)
            Random number generator seed for AUTOQTL. Use this parameter to make sure
            that AUTOQTL will give you the same results each time you run it against the
            same data set with that seed.
        
        config_dict: a Python dictionary or string, optional (default: None)
            Python dictionary:
                A dictionary customizing the operators and parameters that
                TPOT uses in the optimization process.
                For examples, see config_regressor.py 
            Path for configuration file:
                A path to a configuration file for customizing the operators and parameters that
                AUTOQTL uses in the optimization process.
                For examples, see config_regressor.py and config_classifier.py
        
        template: string (default: None)
            Template of predefined pipeline structure. The option is for specifying a desired structure
            for the machine learning pipeline evaluated in AUTOQTL. So far this option only supports
            linear pipeline structure. Each step in the pipeline should be a main class of operators
            (Selector, Transformer or Regressor) or a specific operator
            (e.g. SelectPercentile) defined in AUTOQTL operator configuration. If one step is a main class,
            AUTOQTL will randomly assign all subclass operators (subclasses of SelectorMixin,
            TransformerMixin or RegressorMixin in scikit-learn) to that step.
            Steps in the template are delimited by "-", e.g. "SelectPercentile-Transformer-Regressor".
            By default value of template is None, AUTOQTL generates tree-based pipeline randomly.
        
        warm_start: bool, optional (default: False)
            Flag indicating whether the AUTOQTL instance will reuse the population from
            previous calls to fit().
        
        memory: a Memory object or string, optional (default: None)
            If supplied, pipeline will cache each transformer after calling fit. This feature
            is used to avoid computing the fit transformers within a pipeline if the parameters
            and input data are identical with another fitted pipeline during optimization process.
            String 'auto':
                AUTOQTL uses memory caching with a temporary directory and cleans it up upon shutdown.
            String path of a caching directory
                AUTOQTL uses memory caching with the provided directory and TPOT does NOT clean
                the caching directory up upon shutdown. If the directory does not exist, AUTOQTL will
                create it.
            Memory object:
                AUTOQTL uses the instance of joblib.Memory for memory caching,
                and AUTOQTL does NOT clean the caching directory up upon shutdown.
            None:
                AUTOQTL does not use memory caching.
        
        use_dask: boolean, default False
            Whether to use Dask-ML's pipeline optimizations. This avoid re-fitting
            the same estimator on the same split of data multiple times. It
            will also provide more detailed diagnostics when using Dask's
            distributed scheduler.

            See `avoid repeated work <https://dask-ml.readthedocs.io/en/latest/hyper-parameter-search.html#avoid-repeated-work>`__
            for more details. (NOT USED)
        
        periodic_checkpoint_folder: path string, optional (default: None)
            If supplied, a folder in which AUTOQTL will periodically save pipelines in pareto front so far while optimizing.
            Currently once per generation but not more often than once per 30 seconds.
            Useful in multiple cases:
                Sudden death before AUTOQTL could save optimized pipeline
                Track its progress
                Grab pipelines while it's still optimizing
        
        early_stop: int or None (default: None)
            How many generations AUTOQTL checks whether there is no improvement in optimization process.
            End optimization process if there is no improvement in the set number of generations.
        
        verbosity: int, optional (default: 0)
            How much information AUTOQTL communicates while it's running.
            0 = none, 1 = minimal, 2 = high, 3 = all.
            A setting of 2 or higher will add a progress bar during the optimization procedure.
        
        disable_update_check: bool, optional (default: False)
            Flag indicating whether the AUTOQTL version checker should be disabled.
        
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
        #self.cv = cv
        self.subsample = subsample
        self.n_jobs = n_jobs
        self.max_time_mins = max_time_mins
        self.max_eval_time_mins = max_eval_time_mins
        self.periodic_checkpoint_folder = periodic_checkpoint_folder
        self.early_stop = early_stop
        self.config_dict = config_dict
        self.template = template
        self.warm_start = warm_start
        self.memory = memory
        #self.use_dask = use_dask
        self.verbosity = verbosity
        self.disable_update_check = disable_update_check
        self.random_state = random_state
        self.log_file = log_file

    def _setup_template(self, template):
        """Setup the template for the machine learning pipeline. 
        Accordingly set the minimum and maximum height of the GP tree. AUTOQTL uses the default template, which is None. 
        
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
                        "choose a valid scoring function from the AUTOQTL " 
                        "documentation.".format(scoring)
                    )
                self._scoring_function = scoring # tpot uses the variable name as scoring_function and not _scoring_function but I thought this to be according to code convention.
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
                    self._scoring_function = scoring
    
    def _setup_config(self, config_dict):
        """Setup the configuration dictionary containing the various ML methods, selectors and transformers.
        
        Parameters
        ----------
        config_dict : Python dictionary or string
            custom config dict containing the customizing operators and parameters that AUTOQTL uses in the optimization process
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
                    "custom AUTOQTL operator configuration file: {}".format(e)
                )
        else:
            raise ValueError(
                "Could not open specified AUTOQTL operator config file: "
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
                "{} operators have been imported by AUTOQTL.".format(len(self.operators))
            )

    def _add_operators(self):
        """Add the operators as primitives to the GP primitive set. The operators are in the form of python classes."""

        main_operator_types = ["Regressor", "Selector", "Transformer"] # TPOT uses the variable name main_type
        return_types = [] # TPOT uses the variable name ret_types
        self._op_list = [] # TPOT uses the variable name _op_list

        if self.template == None: # default pipeline structure
            step_in_type = np.ndarray # Input type of each step/operator in the tree
            step_ret_type = Output_Array # Output type of each step/operator in the tree
            
            for operator in self.operators:
                arg_types = operator.parameter_types()[0][1:] # parameter_types() is defined in operator_utils.py, it returns the input and return types of an operator class
                if operator.root:
                    # In case an operator is a root, the return type of that operator can only be Output_Array. In AUTOQTL, a ML method is always the root and cannot exist elsewhere in the tree.
                    tree_primitive_types = ([step_in_type] + arg_types, step_ret_type) # A tuple with input(arguments types) and output type of a root operator
                else:
                    # For a non-root operator the return type is n-dimensional array
                    tree_primitive_types = ([step_in_type] + arg_types, step_in_type)
                
                self._pset.addPrimitive(operator, *tree_primitive_types) # addPrimitive() is a method of the gp.PrimitiveSetTyped(...) class
                self._import_hash_and_add_terminals(operator, arg_types)
            self._pset.addPrimitive(CombineDFs(), [step_in_type, step_in_type], step_in_type)

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
                    self._pset.addPrimitive(
                        CombineDFs(), [step_in_type, step_in_type], step_in_type
                    )
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



