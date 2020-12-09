import pandas as pd
import numpy as np


def load_df_user(file_path):
    header = ['user_id', 'age', 'sex', 'occupation', 'zip_code']
    user_df = pd.read_csv(file_path, sep="|", names=header)
    return user_df


def get_user_data(file_path): # "ml-100k/u.user"
    user_df = load_df_user(file_path)
    user_df = user_df.set_index("user_id")
    user_df["norm_age"] = user_df["age"] / np.max(user_df["age"])
    jobs_onehot = pd.get_dummies(user_df['occupation'], prefix='job')
    user_df["sex_bit"] = [1 if s == "M" else 0 for s in user_df['sex']]
    users_df = pd.concat([user_df, jobs_onehot], axis=1, sort=False)
    users_df = users_df.drop(["age", "sex", "occupation", "zip_code"], axis=1)
    return users_df


def load_df_movie(file_path): #"ml-100k/u.item"
    header = ["movie id", "movie title", "release date", "video release date",
              "IMDb URL", "unknown", "Action", "Adventure", "Animation",
              "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
              "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi",
              "Thriller", "War", "Western"]
    item_df = pd.read_csv(file_path, sep="|", encoding='latin-1', names=header)
    item_df = item_df.drop(["IMDb URL", "video release date", "movie title"], axis=1)
    return item_df


def compute_weighted_ranking(rating_df):
    C = np.sum(rating_df["rating"]) / len(rating_df["rating"])
    m = 3000
    rating_stats = rating_df.groupby("movie_id").agg({"rating": ["mean", "std", "count"]})
    votes = rating_stats["rating"]["count"]
    ratings = rating_stats["rating"]["mean"]
    var = lambda v, R: (v / (v + m)) * R + (m / (v + m)) * C
    rating_stats["wr"] = var(votes, ratings)
    rating_stats["std_rank"] = rating_stats["rating"]["std"] / np.max(rating_stats["rating"]["std"])
    rating_stats["std_rank"] = rating_stats["std_rank"].fillna(0.0)
    return rating_stats


def get_movie_data(file_path, rating_df):
    item_df = load_df_movie(file_path)
    nan_ind = int(item_df.iloc[np.where(item_df["release date"].isna())[0]]["movie id"])

    rating_stats = compute_weighted_ranking(rating_df)
    RATING = pd.DataFrame({"movie_id": rating_stats.index, "wr": np.array(rating_stats[("wr", "")]),
                           "std_rank": np.array(rating_stats[("std_rank", "")])})
    RATING = RATING.set_index("movie_id")

    item_df = item_df.set_index("movie id")
    movies_df = pd.concat([item_df, RATING], axis=1, sort=False)
    movies_df = movies_df.drop(["release date"], axis=1)
    return movies_df, nan_ind


def load_df_ranking(file_path1, file_path2):
    header = ["user_id", "movie_id", "rating", "timestamp"]
    rating_df1 = pd.read_csv(file_path1, sep="\t", names=header)
    rating_df2 = pd.read_csv(file_path2, sep="\t", names=header)
    rating_df = pd.concat([rating_df1, rating_df2], ignore_index=True)
    rating_df = rating_df.reindex()
    rating_df["rating"] = rating_df["rating"] / 5
    return rating_df


def get_seen_movies(rating_df):
    reviewed_movies = {}
    for i, row in rating_df.iterrows():
        user = row["user_id"]
        if user not in reviewed_movies.keys():
            reviewed_movies[user] = {int(row["movie_id"])}
        else:
            reviewed_movies[user].add(int(row["movie_id"]))
    np.save("reviewed_movies.npy", reviewed_movies)
    return reviewed_movies


def get_ranking_data(file_path1, file_path2, file_path1test, file_path2test): #"ml-100k/ua.base", "ml-100k/ub.base""ml-100k/ua.test""ml-100k/ub.test"
    rating_df = load_df_ranking(file_path1, file_path2)
    rating_df_test = load_df_ranking(file_path1test, file_path2test)
    return rating_df, rating_df_test


def clean_nans(rating_df, rating_df_test, nan_ind):
    rating_df = rating_df[rating_df.movie_id != nan_ind]
    rating_df_test = rating_df_test[rating_df_test.movie_id != nan_ind]
    rating_df = rating_df.reset_index()
    rating_df_test = rating_df_test.reset_index()
    return rating_df, rating_df_test


def create_train_label_data(user_df, movies_df, rating_df, stage="train"):
    n_users_cols = len(user_df.columns)
    n_features = len(movies_df.columns) + n_users_cols
    x = np.zeros((len(rating_df), n_features))
    y = np.zeros((len(rating_df), 1))
    for i, row in rating_df.iterrows():
        user = row["user_id"]
        movie = row["movie_id"]
        x[i, 0: n_users_cols] = user_df.loc[user]
        x[i, n_users_cols:] = movies_df.loc[movie]
        y[i] = row["rating"]
    np.save("X{}.npy".format(stage), x)
    np.save("Y{}.npy".format(stage), y)
    return x, y


def preprocessing():
    user_df = get_user_data("ml-100k/u.user")
    rating_df, rating_df_test = get_ranking_data("ml-100k/ua.base", "ml-100k/ub.base",
                                                 "ml-100k/ua.test", "ml-100k/ub.test")
    movies_df, nan_ind = get_movie_data("ml-100k/u.item", rating_df)
    rating_df, rating_df_test = clean_nans(rating_df, rating_df_test, nan_ind)
    reviewed_movies = get_seen_movies(rating_df)

    xtrain, ytrain = create_train_label_data(user_df, movies_df, rating_df, stage="train")
    xtest, ytest = create_train_label_data(user_df, movies_df, rating_df_test, stage="test")

    movies_df.to_csv("movies_df.csv")
    user_df.to_csv("user_df.csv")
    return xtrain, ytrain, xtest, ytest, movies_df, user_df, reviewed_movies

if __name__ == '__main__':
    preprocessing()