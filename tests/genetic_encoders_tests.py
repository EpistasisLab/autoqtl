
"""Testing the working of the genetic encoders defined in the genetic_encoders.py file."""

import sys
import numpy as np

#from autoqtl.builtins.genetic_encoders import DominantEncoder

sys.path.append("C:/Users/ghosha/.vscode/autoqtl")
import autoqtl
from autoqtl.builtins.genetic_encoders import AdditiveEncoder, AdditiveAlternateEncoder, DominantEncoder, RecessiveEncoder, HeterozygoteAdvantageEncoder

X = np.array([[1, 1, 2, 1, 2],
            [1, 0, 0, 1, 2],
            [0, 0, 0, 1, 2],
            [2, 2, 0, 1, 2]])

# Testing AdditiveEncoder operator
def test_AdditiveEncoder():
    """Print that AdditiveEncoder operator returns correct transformed X. """
    X_expected_transformation = np.array([[2, 2, 1, 2, 1],
                                    [2, 3, 3, 2, 1],
                                    [3, 3, 3, 2, 1],
                                    [1, 1, 3, 2, 1]])
    op = AdditiveEncoder()
    #X_transformed = op.transform(X)
    X_transformed = op.fit_transform(X)

    print( np.allclose(X_expected_transformation, X_transformed))

def test_AdditiveEncoder_fit():
    """Assert that fit() in AdditiveEncoder does nothing. """
    op = AdditiveEncoder()
    ret_op = op.fit(X)

    print( ret_op==op )

# Testing AdditiveAlternateEncoder operator
def test_AdditiveAlternateEncoder():
    """Print that AdditiveAlternate operator returns correct transformed X. """
    X_expected_transformation = np.array([[2, 2, 3, 2, 3],
                                        [2, 1, 1, 2, 3],
                                        [1, 1, 1, 2, 3],
                                        [3, 3, 1, 2, 3]])
    op = AdditiveAlternateEncoder()
    X_transformed = op.transform(X)

    print( np.allclose(X_expected_transformation, X_transformed))

def test_AdditiveAlternateEncoder_fit():
    """Assert that fit() in AdditiveAlternateEncoder does nothing. """
    op = AdditiveAlternateEncoder()
    ret_op = op.fit(X)

    print(ret_op==op)

# Testing DominantEncoder operator
def test_DominantEncoder():
    """Print that Dominant operator returns correct transformed X. """
    X_expected_transformation = np.array([[1, 1, 0, 1, 0],
                                         [1, 1, 1, 1, 0],
                                        [1, 1, 1, 1, 0],
                                         [0, 0, 1, 1, 0]])
    op = DominantEncoder()
    X_transformed = op.transform(X)

    print( np.allclose(X_expected_transformation, X_transformed))

def test_DominantEncoder_fit():
    """Assert that fit() in DominantEncoder does nothing. """
    op = DominantEncoder()
    ret_op = op.fit(X)

    print(ret_op==op)

# Testing RecessiveEncoder operator
def test_RecessiveEncoder():
    """Print that Recessive operator returns correct transformed X. """
    X_expected_transformation = np.array([[1, 1, 1, 1, 1],
                                        [1, 0, 0, 1, 1],
                                        [0, 0, 0, 1, 1],
                                        [1, 1, 0, 1, 1]])
    op = RecessiveEncoder()
    X_transformed = op.transform(X)

    print( np.allclose(X_expected_transformation, X_transformed))

def test_RecessiveEncoder_fit():
    """Assert that fit() in ReecessiveEncoder does nothing. """
    op = RecessiveEncoder()
    ret_op = op.fit(X)

    print(ret_op==op)

# Testing HeterorisEncoder operator
def test_HeterorisEncoder():
    """Print that Heterosis operator returns correct transformed X. """
    X_expected_transformation = np.array([[1, 1, 0, 1, 0],
                                        [1, 0, 0, 1, 0],
                                        [0, 0, 0, 1, 0],
                                         [0, 0, 0, 1, 0]])
    op = HeterozygoteAdvantageEncoder()
    X_transformed = op.transform(X)

    print( np.allclose(X_expected_transformation, X_transformed))

def test_HeterosisEncoder_fit():
    """Assert that fit() in HeterosisEncoder does nothing. """
    op = HeterozygoteAdvantageEncoder()
    ret_op = op.fit(X)

    print(ret_op==op)
test_AdditiveEncoder()
test_AdditiveEncoder_fit()

test_AdditiveAlternateEncoder()
test_AdditiveAlternateEncoder_fit()

test_DominantEncoder()
test_DominantEncoder_fit()
    
test_RecessiveEncoder()
test_RecessiveEncoder_fit()

test_HeterorisEncoder()
test_HeterosisEncoder_fit()




