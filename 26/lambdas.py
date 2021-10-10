"""
lambda expressions
"""
square = lambda x: x ** 2
print(square(2))
even = lambda x: x % 2 == 0
print(even(2))
first = lambda s: s[0]
print(first("hello"))
rev = lambda s: s[::-1]
print(rev("hello"))
adder = lambda x, y: x + y
print(adder(2, 3))
