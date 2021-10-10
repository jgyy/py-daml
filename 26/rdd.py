"""
RDD Transformations and Actions
"""
from pyspark import SparkContext

# Examples
sc = SparkContext()
# Show RDD
sc.textFile("example2.txt")
# Save a reference to this RDD
text_rdd = sc.textFile("example2.txt")

# Map a function (or lambda expression) to each line
# Then collect the results.
print(text_rdd.map(lambda line: line.split()).collect())
# Collect everything as a single flat map
print(text_rdd.flatMap(lambda line: line.split()).collect())

# # RDDs and Ksy Value Pairs
services = sc.textFile("services.txt")
print(services.take(2))
print(services.map(lambda x: x[1:] if x[0] == "#" else x).collect())
print(
    services.map(lambda x: x[1:] if x[0] == "#" else x)
    .map(lambda x: x.split())
    .collect()
)

# Using Key Value Pairs for Operations
cleanServ = services.map(lambda x: x[1:] if x[0] == "#" else x).map(lambda x: x.split())
print(cleanServ.collect())
# Let's start by practicing grabbing fields
clean_serv2 = cleanServ.map(lambda lst: (lst[3], lst[-1])).collect()
print(clean_serv2)
# Continue with reduceByKey
# Notice how it assumes that the first item is the key!
clean_serv3 = cleanServ.map(lambda lst: (lst[3], lst[-1])).reduceByKey(
    lambda amt1, amt2: amt1 + amt2
).collect()
print(clean_serv3)
# Continue with reduceByKey
# Notice how it assumes that the first item is the key!
clean_serv4 = cleanServ.map(lambda lst: (lst[3], lst[-1])).reduceByKey(
    lambda amt1, amt2: float(amt1) + float(amt2)
).collect()
print(clean_serv4)

# Grab state and amounts Add them
# Get rid of ('State','Amount')
# Sort them by the amount value
clean_serv5 = cleanServ.map(lambda lst: (lst[3], lst[-1])).reduceByKey(
    lambda amt1, amt2: float(amt1) + float(amt2)
).filter(lambda x: not x[0] == "State").sortBy(
    lambda stateAmount: stateAmount[1], ascending=False
).collect()
print(clean_serv5)

x = ["ID", "State", "Amount"]
func1 = lambda x: x[-1]
func2 = lambda x: x[-1]
print(func1(x))
print(func2(x))
