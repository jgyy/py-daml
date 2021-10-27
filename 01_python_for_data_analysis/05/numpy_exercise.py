"""
NumPy Exercises - Solutions
"""
from numpy import zeros, ones, arange, eye, linspace
from numpy.random import rand, randn


def wrapper():
    """
    wrapper function
    """
    print(zeros(10))
    print(ones(10))
    print(ones(10) * 5)
    print(arange(10, 51))
    print(arange(10, 51, 2))
    print(arange(9).reshape(3, 3))
    print(eye(3))
    print(rand(1))
    print(randn(25))
    print(arange(1, 101).reshape(10, 10) / 100)
    print(linspace(0, 1, 20))
    mat = arange(1, 26).reshape(5, 5)
    print(mat)
    print(mat[2:, 1:])
    print(mat[3, 4])
    print(mat[:3, 1:2])
    print(mat[4, :])
    print(mat[3:5, :])
    print(mat.sum())
    print(mat.std())
    print(mat.sum(axis=0))


if __name__ == "__main__":
    wrapper()
