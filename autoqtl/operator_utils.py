"""This file contains the class, class medthod and function definitions required to generate the Genetic Programming operators. """

from pydoc import classname
import numpy as np
from sklearn.base import BaseEstimator, is_regressor
from sklearn.gaussian_process.kernels import Kernel
import inspect

class Operator(object):
    """Base class for operators in AutoQTL. class object is passed as a parameter to make the class a new style python class for older versions. """

    root = False # Whether this operator type can be the root of the genetic tree
    import_hash = None # Specifies the packages required to be imported to use the operator
    package_class = None # Specifies the class (part of sklearn package or AutoQTL package) which contains the functionalities of the operator. 
    arg_types = None # Types of the argument of the operator


class ARGType(object):
    """Base class for argument/parameter specifications. """

    pass

def source_decode(sourcecode, verbose=0):
    """Decode operator source and import operator class.
    
    Parameters
    ----------
    sourcecode : string
        a string of operator source, the entire path of the source from which the operator is imported (e.g 'sklearn.feature_selection.SelectPerecentile')
        
    verbose : int, optional (default: 0)
        How much information AUTOQTL package communicates while it's running. 
        0 = none, 1 = minimal, 2 = high, 3 = all.
        if verbose > 2 then ImportError will raise during initialization
        
    Returns
    -------
    import_str : string
        a string of operator class source (e.g. 'sklearn.feature_selection')
    
    op_str : string
        a string of operator class (e.g 'SelectPercentile')
    
    op_obj : object
        operator class object (e.g SelectPercentile)
    
    """
    tmp_path = sourcecode.split(".")
    op_str = tmp_path.pop()
    import_str = ".".join(tmp_path) # After the operator class string is popped, the rest of the path is again joined with a "."
    try:
        if sourcecode.startswith("autoqtl."):
            exec("from {} import {}".format(import_str[7:], op_str)) # When importing the operator class from AutoQTL package, as already in the package location hence omitting the 'autoql' part of the path
        else:
            exec("from {} import {}".format(import_str, op_str))
        op_obj = eval(op_str)
    except Exception as e:
        if verbose > 2:
            raise ImportError("Error: could not import {}.\n{}".format(sourcecode, e))
        else:
            print(
                "Warning: {} is not available and will not be used by AUTOQTL.".format(
                    sourcecode
                )
            )
        op_obj = None
    return import_str, op_str, op_obj

def set_sample_weight(pipeline_steps, sample_weight=None):
    """Recursively iterates through all objects in the pipeline and sets sample weight.
    
    Parameters
    ----------
    pipeline_steps : array-like
        List of (str, obj) tuples from a scikit-learn pipeline or related object
        
    sample_weight : array-like
        List of sample weight
        
    Returns
    -------
    sample_weight_dict :
        A dictionary of sample_weight
        
    """
    sample_weight_dict = {}
    if not isinstance(sample_weight, type(None)):
        for (pname, obj) in pipeline_steps:
            if inspect.getfullargspec(obj.fit).args.count("sample_weight"):
                step_sw = pname + "_sample_weight"
                sample_weight_dict[step_sw] = sample_weight
    
    if sample_weight_dict:
        return sample_weight_dict
    else:
        return None

def _is_selector(estimator):
    """Checks if an estimator is of type selector.
    
    Parameter
    ---------
    estimator : object
        the class object for the operator
    
    Returns
    -------
    bool value : True if estimator is a selector else false
    
    """
    selector_methods = [
        "get_support",
        "transform",
        "inverse_transform",
        "fit_transform",
    ]
    return all(hasattr(estimator, method) for method in selector_methods)

def _is_transformer(estimator):
    """Checks if an estimator is of type transformer.
    
    Parameter
    ---------
    estimator : object
        the class object for the operator
    
    Returns
    -------
    bool value : True if estimator is a transformer else false
    """
    return hasattr(estimator, "fit_transform")

def ARGTypeClassFactory(classname, prange, BaseClass=ARGType):
    """Dynamically create parameter type class.
    
    Parameters
    ----------
    classname : string
        parameter name in a operator
    
    prange : list
        list of values for the parameter in a operator

    BaseClass : Class
        inherited BaseClass (AUTOQTL class) for parameter

    Returns 
    -------

    Class
        parameter class

    """
    return type(classname, (BaseClass,), {"values": prange})

def AUTOQTLOperatorClassFactory(
    opsource, opdict, BaseClass=Operator, ArgBaseClass=ARGType, verbose=0
):
    """Dynamically create Operator class.
    
    Parameters
    ----------
    opsource : string
        operator source in config dictionary (key)
        
    opdict : dictionary
        operator parameters in config dictionary (value)
    
    regression : bool
        True if it can be used in Regression problem
        
    BaseClass : Class
        inherited BaseClass for operator
        
    ArgBaseClass : Class
        inherited BaseClass for parameter
        
    verbose : int, optional (default : 0)
        How much information AUTOQTL package communicates while it's running.
        0 = none, 1 = minimal, 2 = high, 3 = all.
        if verbose > 2 then ImportError will raise during initialization
        
    Returns
    -------
    op_class : Class
        a new class for a operator
    
    arg_types : list
        a list of parameter class
        
    """
    class_profile = {} # dictionary to hold the features of the operator class
    dep_op_list = {} # list of nested estimator/callable function (parameter dictionary level), contains the operator class string
    dep_op_type = {} # type of nested estimator/callable function (parameter dictionary level), contains the operator class object

    import_str, op_str, op_obj = source_decode(opsource, verbose=verbose) # the operator source is split to get the required information

    if not op_obj:
        return None, None
    
    else:
        # define if the operator can be the root of a pipeline

        if is_regressor(op_obj):
            class_profile["root"] = True
            optype = "Regressor"
        
        elif _is_selector(op_obj):
            optype = "Selector"

        elif _is_transformer(op_obj):
            optype = "Transformer"

        else:
            raise ValueError(
                "optype (Operator Type) must be one of: Regressor, Selector, Transformer"
            ) 
        
        @classmethod
        def op_type(cls):
            """Return the operator type.
            
            Possible values:
                "Regressor", "Selector", "Transformer"
            
            """
            return optype
        
        class_profile["type"] = op_type
        class_profile["package_class"] = op_obj
        import_hash = {} # dictionary to contain the details of the required class to be imported from a pakage
        import_hash[import_str] = [op_str]
        arg_types = [] # list of argument type class (to be returned)

        for pname in sorted(opdict.keys()): # iterates through each parameter present in the opdict
            prange = opdict[pname]
            
            if not isinstance(prange, dict): # if the poosible values for the parameter is not in a nested dictionary 
                classname = "{}__{}".format(op_str, pname) # e.g SelectVariance__threshold
                arg_types.append(ARGTypeClassFactory(classname, prange, ArgBaseClass)) # appened to the list of argument classes
            
            else: # nested dictionary exists
                for dkey, dval in prange.items():
                    dep_import_str, dep_op_str, dep_op_obj = source_decode(
                        dkey, verbose=verbose
                    ) # the operator (parameter for a parameter, e.g score_func = f_regression) in the nested dictionary is split to get the required information
                    if dep_import_str in import_hash: # if the package required to import the current operator class is already present in the import_hash dict, just append the operator class
                        import_hash[dep_import_str].append(dep_op_str) 
                    else:
                        import_hash[dep_import_str] = [dep_op_str] # if a new package is required to import the operator class, add it to the import_hash dictionary
                    
                    dep_op_list[pname] = dep_op_str
                    dep_op_type[pname] = dep_op_obj

                    if dval: # if the values of the nested dictionary is not None
                        for dpname in sorted(dval.keys()):
                            dprange = dval[dpname]
                            classname = "{}__{}__{}".format(op_str, dep_op_str, dpname) # e.g SelectPercentile_score_func_f_regression
                            arg_types.append(ARGTypeClassFactory(classname, dprange, ArgBaseClass))
        
        class_profile["arg_types"] = tuple(arg_types)
        class_profile["import_hash"] = import_hash
        class_profile["dep_op_list"] = dep_op_list
        class_profile["dep_op_type"] = dep_op_type
        
        @classmethod
        def parameter_types(cls):
            """Return the argument and return types of an operator.
            
            Parameters
            ----------
            None
            
            Returns
            -------
            parameter_types : tuple
                Tuple of the DEAP parameter types and the DEAP return type for the operator
                
            """
            return ([np.ndarray] + arg_types, np.ndarray) # (input types, return types)
        
        class_profile["parameter_types"] = parameter_types

        @classmethod
        def export(cls, *args):
            """Represent the operator as a string so that it can be exported to a file.

            Parameters
            ----------
            args
                Arbitrary arguments to be passed to the operator

            Returns
            -------
            export_string: str
                String representation of the sklearn class with its parameters in
                the format:
                SklearnClassName(param1="val1", param2=val2)

            """
            op_arguments = []

            if dep_op_list:
                dep_op_arguments = {}
                for dep_op_str in dep_op_list.values():
                    dep_op_arguments[dep_op_str] = []

            for arg_class, arg_value in zip(arg_types, args):
                aname_split = arg_class.__name__.split("__")
                if isinstance(arg_value, str):
                    arg_value = '"{}"'.format(arg_value)
                if len(aname_split) == 2:  # simple parameter
                    op_arguments.append("{}={}".format(aname_split[-1], arg_value))
                # Parameter of internal operator as a parameter in the
                # operator, usually in Selector
                else:
                    dep_op_arguments[aname_split[1]].append(
                        "{}={}".format(aname_split[-1], arg_value)
                    )

            tmp_op_args = []
            if dep_op_list:
                # To make sure the inital operators is the first parameter just
                # for better persentation
                for dep_op_pname, dep_op_str in dep_op_list.items():
                    arg_value = dep_op_str  # a callable function, e.g scoring function
                    doptype = dep_op_type[dep_op_pname]
                    if inspect.isclass(doptype):  # a estimator
                        if (
                            issubclass(doptype, BaseEstimator)
                            or is_regressor(doptype)
                            or _is_transformer(doptype)
                            or issubclass(doptype, Kernel)
                        ):
                            arg_value = "{}({})".format(
                                dep_op_str, ", ".join(dep_op_arguments[dep_op_str])
                            )
                    tmp_op_args.append("{}={}".format(dep_op_pname, arg_value))
            op_arguments = tmp_op_args + op_arguments
            return "{}({})".format(op_obj.__name__, ", ".join(op_arguments))

        class_profile["export"] = export

        op_classname = "AUTOQTL_{}".format(op_str)
        op_class = type(op_classname, (BaseClass,), class_profile)
        op_class.__name__ = op_str
        
        return op_class, arg_types



    