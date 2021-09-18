"""
Recommender Systems
"""
from warnings import filterwarnings
from pandas import read_csv, DataFrame, merge
from seaborn import set_style, jointplot
from matplotlib.pyplot import figure, show

# Get the Data
filterwarnings("ignore", category=RuntimeWarning)
column_names = ["user_id", "item_id", "rating", "timestamp"]
df = DataFrame(read_csv("u.data", sep="\t", names=column_names))
print(df.head())
movie_titles = DataFrame(read_csv("Movie_Id_Titles"))
print(movie_titles.head())
df = merge(df, movie_titles, on="item_id")
print(df.head())

# EDA
set_style("white")
print(df.groupby("title")["rating"].mean().sort_values(ascending=False).head())
print(df.groupby("title")["rating"].count().sort_values(ascending=False).head())
ratings = DataFrame(df.groupby("title")["rating"].mean())
print(ratings.head())
ratings["num of ratings"] = DataFrame(df.groupby("title")["rating"].count())
print(ratings.head())

# Plots
figure(figsize=(10, 4))
ratings["num of ratings"].hist(bins=70)
figure(figsize=(10, 4))
ratings["rating"].hist(bins=70)
jointplot(x="rating", y="num of ratings", data=ratings, alpha=0.5)

# Recommending Similar Movies
moviemat = df.pivot_table(index="user_id", columns="title", values="rating")
print(moviemat.head())
print(ratings.sort_values("num of ratings", ascending=False).head(10))
print(ratings.head())
starwars_user_ratings = moviemat["Star Wars (1977)"]
liarliar_user_ratings = moviemat["Liar Liar (1997)"]
print(starwars_user_ratings.head())

# Use corrwith() method to get correlations between two pandas series
similar_to_starwars = moviemat.corrwith(starwars_user_ratings)
similar_to_liarliar = moviemat.corrwith(liarliar_user_ratings)
corr_starwars = DataFrame(similar_to_starwars, columns=["Correlation"])
corr_starwars.dropna(inplace=True)
print(corr_starwars.head())
print(corr_starwars.sort_values("Correlation", ascending=False).head(10))
corr_starwars = corr_starwars.join(ratings["num of ratings"])
print(corr_starwars.head())
print(
    corr_starwars[corr_starwars["num of ratings"] > 100]
    .sort_values("Correlation", ascending=False)
    .head()
)
corr_liarliar = DataFrame(similar_to_liarliar, columns=["Correlation"])
corr_liarliar.dropna(inplace=True)
corr_liarliar = corr_liarliar.join(ratings["num of ratings"])
print(
    corr_liarliar[corr_liarliar["num of ratings"] > 100]
    .sort_values("Correlation", ascending=False)
    .head()
)

# LAST STEP
show()
