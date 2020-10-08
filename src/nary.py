from math import floor, log
from typing import Generator, List

def to_nary(x: int, n: int) -> List[int]:
    """
    Transform an integer x into n-ary number, which is represented by a list of integers.

    Argument:
    x: {int} - An integer transformed into n-ary
    n: {int} - A base number

    Returns:
    {List[int]} - Representation of x as an n-ary number
    """
    return [*reversed([(x // (n ** i)) % n for i in range(floor(log(x, n))+1)])] if x else [0]


def nary_with_length(x: int, n: int, length: int) -> List[int]:
    """
    Represent an integer x as an n-ary number with given length.

    Argument:
    x: {int} - An integer transformed into n-ary
    n: {int} - A base number
    length: {int} - A length of a list representing n-ary of x

    Raises:
    ValueError -- When the given length is less than floor(log(x, n))+1.

    Returns:
    {List[int]} - Representation of x as an n-ary number with given length
    """
    digit = floor(log(x, n))+1 if x else 1
    if length < digit:
        raise ValueError("Invalid length: The length must be at most floor(log(x, n))+1.")
    return [0] * (length - digit) + to_nary(x, n)


def all_nary(n: int, length: int) -> Generator[List[int], None, None]:
    """
    Generate all n-ary numbers formatted as a list of integers.

    Arguments:
    length: {int} - The length of a list representing a p-ary number.
    n     : {int} - Base number.

    Yields:
    {List[int]} - An expression of n-ary number as a list of integers.
    """
    for i in range(n ** length):
        yield nary_with_length(i, n, length)