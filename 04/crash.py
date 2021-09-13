"""
Python Crash Course
"""
# Numbers
print(1 + 1)
print(1 * 3)
print(1 / 2)
print(2 ** 4)
print(4 % 2)
print(5 % 2)
print((2 + 3) * (5 + 5))

# Variable Assignment
VAR = 2
X = 2
Y = 3
Z = X + Y
print(VAR, X, Y, Z)

# Strings
print("double quotes")
print(" wrap lot's of other quotes")

# Printing
X = "hello"
print(X)
NUM = 12
NAME = "Sam"
print(f"My number is {NUM}, and my name is: {NAME}.")

# Lists
print([1, 2, 3])
print(["hi", 1, [1, 2]])
my_list = ["a", "b", "c"]
my_list.append("d")
print(my_list)
print(my_list[0])
print(my_list[1])
print(my_list[1:])
print(my_list[:1])
my_list[0] = "NEW"
print(my_list)
nest = [1, 2, 3, [4, 5, ["target"]]]
print(nest[3])
print(nest[3][2])
print(nest[3][2][0])

# Dictionaries
d = {"key1": "item1", "key2": "item2"}
print(d)
print(d["key1"])

# Booleans
print(True)
print(False)

# Tuples
t = (1, 2, 3)
print(t[0])

# Sets
print({1, 2, 3})
print({1, 2, 3, 1, 2, 1, 2, 3, 3, 3, 3, 2, 2, 2, 1, 1, 2})

# Comparison Operators
print(1 > 2)
print(1 < 2)
print(1 <= 4)
print("hi" == "bye")

# Logical Operators
print((1 > 2) and (2 < 3))
print((1 > 2) or (2 < 3))
print((1 == 2) or (2 == 3))

# if elif else Statements
if 1 < 2:
    print("Yep!")
if 1 < 2:
    print("yep!")
if 1 < 2:
    print("first")
else:
    print("last")
if 1 == 2:
    print("first")
elif 3 == 4:
    print("middle")
else:
    print("Last")

# For Loops
seq = [1, 2, 3, 4, 5]
for item in seq:
    print(item)
for item in seq:
    print("Yep")
for jelly in seq:
    print(jelly + jelly)

# While Loops
i = 1
while i < 5:
    print(f"i is {i}")
    i += 1

# range()
print(range(5))
for i in range(5):
    print(i)
print(list(range(5)))

# list comprehension
x = [1, 2, 3, 4]
OUT = []
for item in x:
    OUT.append(item ** 2)
print(OUT)

print([item ** 2 for item in x])

# functions
def my_func(param1="default"):
    """
    Docstring goes here.
    """
    print(param1)


print(my_func)
my_func()
my_func("new param")
my_func(param1="new param")


def square(numx):
    """
    returns number squared
    """
    return numx ** 2


OUT = square(2)
print(OUT)

# lambda expressions
def times2(var):
    """
    value multiplied by 2
    """
    return var * 2


print(times2(2))
print(lambda var: var * 2)

# map and filter
seq = [1, 2, 3, 4, 5]
print(map(times2, seq))
print(list(map(times2, seq)))
print(list(map(lambda var: var * 2, seq)))
print(list(filter(lambda var: var % 2 == 0, seq)))

# methods
ST = "hello my name is Sam"
print(ST.lower())
print(ST.upper())
print(ST.split())
TWEET = "Go Sports! #Sports"
print(TWEET.split("#"))
print(TWEET.split("#")[1])
print(d)
print(d.keys())
print(d.items())
lst = [1, 2, 3]
print(lst.pop())
print(lst)
print("x" in [1, 2, 3])
print("x" in ["x", "y", "z"])
