"""
Numpy Array Indexing
"""
from numpy import arange, array, zeros

# Creating sample array
arr = arange(0, 11)
print(arr)

# Get a value at an index
print(arr[8])
print(arr[1:5])

# Setting a value with index range (Broadcasting)
arr[0:5] = 100
print(arr)

# Reset array, we'll see why I had to reset in  a moment
arr = arange(0, 11)
print(arr)

# Important notes on Slices
slice_of_arr = arr[0:6]
print(slice_of_arr)

# Change Slice
slice_of_arr[:] = 99
print(slice_of_arr)
print(arr)

# To get a copy, need to be explicit
arr_copy = arr.copy()
print(arr_copy)

# Indexing a 2D array
arr_2d = array(([5, 10, 15], [20, 25, 30], [35, 40, 45]))
print(arr_2d)
print(arr_2d[1])
print(arr_2d[1][0])
print(arr_2d[1, 0])
print(arr_2d[:2, 1:])
print(arr_2d[2])
print(arr_2d[2, :])

# Set up matrix
arr2d = zeros((10, 10))
arr_length = arr2d.shape[1]
for i in range(arr_length):
    arr2d[i] = i
print(arr2d)
print(arr2d[[2, 4, 6, 8]])
print(arr2d[[6, 4, 2, 7]])

# Selection
arr = arange(1, 11)
print(arr)
bool_arr = arr > 4
print(bool_arr)
print(arr[bool_arr])
print(arr[arr > 2])
X = 2
print(arr[arr > X])
