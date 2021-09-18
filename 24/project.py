"""
Natural Language Processing Project
"""
from pandas import read_csv, DataFrame
from seaborn import set_style, FacetGrid, boxplot, countplot, heatmap
from matplotlib.pyplot import show, hist, figure
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline

# The Data
yelp = DataFrame(read_csv("yelp.csv"))
print(yelp.head())
print(yelp.info())
print(yelp.describe())
yelp["text length"] = yelp["text"].apply(len)

# EDA
set_style("white")
g = FacetGrid(yelp, col="stars")
g.map(hist, "text length")
figure()
boxplot(x="stars", y="text length", data=yelp, palette="rainbow")
figure()
countplot(x="stars", data=yelp, palette="rainbow")
stars = yelp.groupby("stars").mean()
print(stars)
print(stars.corr())
figure()
heatmap(stars.corr(), cmap="coolwarm", annot=True)

# NLP Classification Task
yelp_class = yelp[(yelp.stars == 1) | (yelp.stars == 5)]
X = yelp_class["text"]
y = yelp_class["stars"]
cv = CountVectorizer()
X = cv.fit_transform(X)

# Train Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)

# Training a Model
nb = MultinomialNB()
nb.fit(X_train, y_train)

# Predictions and Evaluations
predictions = nb.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# Using Text Processing
pipeline = Pipeline([("bow", CountVectorizer()), ("classifier", MultinomialNB())])

# Train Test Split
X = yelp_class["text"]
y = yelp_class["stars"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=101
)
# May take some time
pipeline.fit(X_train, y_train)

# Prediction and Evaluation
predictions = pipeline.predict(X_test)
print(confusion_matrix(y_test, predictions))
print(classification_report(y_test, predictions))

# LAST STEP
show()
