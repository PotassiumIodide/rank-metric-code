import numpy as np
import random
from typing import List

from .rankmetric import wt
from .type_def import Matrix, prime

def show_matrix(matrix: Matrix) -> str:
    """
    Give a string-type expression of a given matrix.

    Arguments:
    matrix: {numpy.ndarray} - a matrix

    Returns:
    {str} - A string-type expression of a given matrix
    """
    return "".join(["[", "\n ".join([str(row) for row in list(matrix)]), "]"])


def show_matrices(matrices: List[Matrix]) -> str:
    """
    Give a string-type expression of a set of matrices.
    Since numpy.ndarray is not immutable, admit a list of matrices instead.

    Argument:
    matrices: {List[numpy.ndarray]} - A set of matrices

    Returns:
    {str} - A string-type expression of a set of matrices.
    """
    return "\n".join(["{", ",\n".join(map(show_matrix, matrices)), "}"])


def random_nonzero_matrix(rows: int, cols: int, p: prime) -> np.ndarray:
    """
    Randomly choose a (rows × cols) non-zero matrix over Fp

    Arguments:
    rows: {int} - The number of row vectors in the returned matrix
    cols: {int} - The number of column vectors in the returned matrix
    p   : {int} - A prime number for a field Fp

    Returns:
    {numpy.ndarray} - A random (rows × cols) non-zero matrix over Fp
    """
    c = np.array( [ random.randint(0, p-1) for _ in range(rows*cols) ] ).reshape(rows, cols) % p
    return c if wt(c) else random_nonzero_matrix(rows, cols, p)


def random_nonzero_matrices(rows: int, cols: int, num: int, p: prime) -> List[np.ndarray]:
    """
    Randomly choose (rows × cols) non-zero matrices with no duplication

    Arguments:
    rows: {int} - The number of row vectors in a random matrix
    cols: {int} - The number of column vectors in a random matrix
    num : {int} - The number of returned matrices
    p   : {int} - A prime number for a field Fp

    Returns:
    {List[numpy.ndarray]} - A set of random (rows × cols) non-zero matrices over Fp
    """
    return [
        *map(
            lambda t: np.array(t).reshape(rows, cols),
            list({tuple(np.mod((random_nonzero_matrix(rows, cols, p)), p).flatten()) for _ in range(num)})
        )
    ]