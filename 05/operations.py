"""
Numpy Operations
"""
from numpy import arange, sqrt, exp, sin, log

# Arithmetic
arr = arange(1, 10)
print(arr + arr)
print(arr * arr)
print(arr - arr)
print(arr / arr)
print(1 / arr)
print(arr ** 3)

# Universal Array functions
print(sqrt(arr))
print(exp(arr))
print(arr.max())
print(sin(arr))
print(log(arr))
