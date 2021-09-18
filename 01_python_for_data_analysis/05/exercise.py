"""
Numpy Exercises Overview
"""
from numpy.random import randn
from numpy import zeros, ones, arange, eye, linspace, std

# Create an array of 10 zeros
print(zeros(10))
print(ones(10))
print(ones(10) * 5)
print(arange(10, 51))
print(arange(10, 51, 2))
print(arange(9).reshape(3, 3))
print(arange(9).reshape(3, 3))
print(eye(3))
print(randn(1))
print(randn(25))
print(arange(0.01, 1.01, 0.01).reshape(10, 10))
print(linspace(0.01, 1, 20))

# Numpy Indexing and Selection
mat = arange(1,26).reshape(5,5)
print(mat)
print(mat[2:, 1:])
print(mat[3, 4])
print(mat[:3, 1:2])
print(mat[4, :])
print(mat[3:, :])
print(mat.sum())
print(std(mat))
print(mat.sum(axis=0))
