"""This file contains the class, class medthod and function definitions required to generate the Genetic Programming operators. """

import numpy as np
from sklearn.base import BaseEstimator, is_regressor
import inspect

class Operator(object):
    """Base class for operators in AUTOQTL. class object is passed as a parameter to make the class a new style python class for older versions. """

    root = False # Whether this operator type can be the root of the genetic tree
    import_hash = None # Specifies the packages required to be imported to use the operator
    package_class = None # Specifies the class (part of sklearn package or AUTOQTL package) which contains the functionalities of the operator. Renamed from sklearn_class (TPOT)
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
            exec("from {} import {}".format(import_str[7:], op_str)) # When importing the operator class from AUTOQTL package, as already in the package location hence omitting the 'autoql' part of the path
        else:
            exec("from {} import {}".format(import_str, op_str))
        op_obj = eval(op_str)
    except Exception as e:
        if verbose > 2:
            raise ImportError("Error: could not import {}.\n{}".format(sourcecode, e))
        else:
            print(
                "Warning: {} is nor available and will not be used by TPOT.".format(
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

