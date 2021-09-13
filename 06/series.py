"""
Series
"""
from numpy import array
from pandas import Series

# Creating a Series
labels = ["a", "b", "c"]
my_list = [10, 20, 30]
arr = array([10, 20, 30])
d = {"a": 10, "b": 20, "c": 30}
print(Series(my_list))
print(Series(my_list, labels))
print(Series(arr))
print(Series(d))
print(Series(labels))
print(Series([sum, print, len]))

# Using an Index
ser1 = Series([1, 2, 3, 4], ["USA", "Germany", "USSR", "Japan"])
print(ser1)
ser2 = Series([1, 2, 5, 4], index=["USA", "Germany", "Italy", "Japan"])
print(ser2)
print(ser1["USA"])
print(ser1 + ser2)
