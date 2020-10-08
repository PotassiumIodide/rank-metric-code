import numpy as np
from typing import List

from .nary import all_nary
from .type_def import prime

def F(p: prime, n: int) -> List[np.ndarray]:
    """
    Return the list of all n-dimensional vectors in a field Fp.

    Arguments:
    p: {int} - prime number
    n: {int} - the length (or dimension) of vector

    Returns:
    {List[numpy.array]} - The list of all n-dimensional vectors in Fp
    """
    return [np.array(v) for v in all_nary(p, n)]