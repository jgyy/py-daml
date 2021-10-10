"""
Introduction to Spark and Python
"""
from pyspark import SparkContext

# Basic Operations
sc = SparkContext()
textFile = sc.textFile("example.txt")
print(textFile.count())
print(textFile.first())

# Transformations
secfind = textFile.filter(lambda line: "second" in line)
# RDD
print(secfind.count())
print(secfind.count())
print(secfind.count())
