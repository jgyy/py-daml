"""
Numpy Arrays
"""
from numpy.random import randn, randint
from numpy import array, arange, zeros, ones, linspace, eye

# array
my_list = [1, 2, 3]
print(my_list)
print(array(my_list))
my_matrix = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
print(my_matrix)
print(array(my_matrix))

# arange
print(arange(0, 10))
print(arange(0, 11, 2))

# zeros and ones
print(zeros(3))
print(zeros((5, 5)))
print(ones(3))
print(ones((3, 3)))

# linspace
print(linspace(0, 10, 3))
print(linspace(0, 10, 50), 1)

# eye
print(eye(4))

# random
print(randn(2))
print(randn(5, 5))
print(randint(1, 100))
print(randint(1, 100, 10))

# Array Attrubutes and Methods
arr = arange(25)
ranarr = randint(0, 50, 10)
print(arr)
print(ranarr)
print(arr.reshape(5, 5))
print(ranarr.max())
print(ranarr.argmax())
print(arr.shape)
print(arr.reshape(1, 25).shape)
print(arr.reshape(25, 1).shape)
print(arr.dtype)
