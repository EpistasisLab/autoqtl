"""Testing the working of the genetic encoders defined in the genetic_encoders.py file."""

import numpy as np
from autoqtl.builtins import AdditiveEncoder

X = np.array([[1, 1, 2, 1, 2],
            [1, 0, 0, 1, 2],
            [0, 0, 0, 1, 2],
            [2, 2, 0, 1, 2]])

def test_AdditiveEncoder():
    """Assert that AdditiveEncoder operator returns correct transformed X. """
    X_correct_transform = np.array([[2, 2, 1, 2, 1],
                                    [2, 3, 3, 2, 1],
                                    [3, 3, 3, 2, 1],
                                    [1, 1, 3, 2, 1]])
    obj = AdditiveEncoder()
    X_transformed = obj.transform(X)

    assert np.allclose(X_correct_transform, X_transformed)

    





