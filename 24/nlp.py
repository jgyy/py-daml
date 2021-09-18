"""
Natural Language Processing
"""
from io import StringIO
from re import sub
from string import punctuation
from pandas import read_csv, DataFrame
from matplotlib.pyplot import figure, show
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

# Get the Data manually
with open("SMSSpamCollection", encoding="utf-8") as lines:
    messages = [line.rstrip() for line in lines]
print(len(messages))
for message_no, message in enumerate(messages[5:10]):
    print(message_no + 5, message)

# Get the Data with pandas
messages = DataFrame(
    read_csv("SMSSpamCollection", sep="\t", names=["label", "message"])
)
print(messages.head())

# Exploratory Data Analysis
print(messages.describe())
print(messages.groupby("label").describe())
messages["length"] = messages["message"].apply(len)
print(messages.head())

# Data Visualization
figure()
messages["length"].plot(bins=50, kind="hist")
print(messages.length.describe())
print(messages[messages["length"] == 910]["message"].iloc[0])
messages.hist(column="length", by="label", bins=50, figsize=(12, 4))

# Text Pre-processing
MESS = "Sample message! Notice: it has punctuation."
# Check characters to see if they are in punctuation
NOPUNC = [char for char in MESS if char not in punctuation]
# Join the characters again to form the string.
NOPUNC = "".join(NOPUNC)
print(NOPUNC)
# Show some stop words
print(stopwords.words("english")[0:10])
print(NOPUNC.split())
# Now just remove any stopwords
clean_mess = [
    word for word in NOPUNC.split() if word.lower() not in stopwords.words("english")
]
print(clean_mess)


def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in punctuation]
    # Join the characters again to form the string.
    nopunc = "".join(nopunc)
    # Now just remove any stopwords
    return [
        word
        for word in nopunc.split()
        if word.lower() not in stopwords.words("english")
    ]


print(messages.head())
# Check to make sure its working
print(messages["message"].head(5).apply(text_process))
# Show original dataframe
print(messages.head())

# Vectorisation Might take awhile...
bow_transformer = CountVectorizer(analyzer=text_process).fit(messages["message"])
# Print total number of vocab words
print(len(bow_transformer.vocabulary_))
message4 = messages["message"][3]
print(message4)
bow4 = bow_transformer.transform([message4])
df = DataFrame(
    read_csv(
        StringIO(sub(r"\(0, |\)", "", str(bow4))),
        delimiter="\t",
        names=["index", "count"],
    )
)
print(bow4)
print(bow4.shape)
print(df[df["count"] == 2]["index"])
for i in df[df["count"] == 2]["index"]:
    print(bow_transformer.get_feature_names()[int(i)])
messages_bow = bow_transformer.transform(messages["message"])
print("Shape of Sparse Matrix: ", messages_bow.shape)
print("Amount of Non-Zero occurences: ", messages_bow.nnz)
sparsity = 100.0 * messages_bow.nnz / (messages_bow.shape[0] * messages_bow.shape[1])
print("sparsity: {}".format(round(sparsity)))

# TF-IDF
tfidf_transformer = TfidfTransformer().fit(messages_bow)
tfidf4 = tfidf_transformer.transform(bow4)
print(tfidf4)
print(tfidf_transformer.idf_[bow_transformer.vocabulary_["u"]])
print(tfidf_transformer.idf_[bow_transformer.vocabulary_["university"]])
messages_tfidf = tfidf_transformer.transform(messages_bow)
print(messages_tfidf.shape)

# Training a model
spam_detect_model = MultinomialNB().fit(messages_tfidf, messages["label"])
print("predicted:", spam_detect_model.predict(tfidf4)[0])
print("expected:", messages.label[3])

# Model Evaluation
all_predictions = spam_detect_model.predict(messages_tfidf)
print(all_predictions)
print(classification_report(messages["label"], all_predictions))

# Train Test Split
msg_train, msg_test, label_train, label_test = train_test_split(
    messages["message"], messages["label"], test_size=0.2
)
print(len(msg_train), len(msg_test), len(msg_train) + len(msg_test))

# Creating a Data Pipeline
pipeline = Pipeline(
    [
        ("bow", CountVectorizer(analyzer=text_process)),
        ("tfidf", TfidfTransformer()),
        ("classifier", MultinomialNB()),
    ]
)
pipeline.fit(msg_train, label_train)
predictions = pipeline.predict(msg_test)
print(classification_report(predictions, label_test))

# LAST STEP
show()
