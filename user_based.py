import numpy as np
import pandas as pd


def read_file(filename):
    f = open(filename, 'r')
    ratings, users, movies = [], [], []
    while True:
        line = f.readline()
        if line == '':
            break
        user, num = line.split('|')
        users.append(int(user))
        for i in range(int(num)):
            line = f.readline()
            item, rating = line.split('  ')
            movies.append(int(item))
            ratings.append([int(user), int(item), int(rating)])
    users = np.unique(users)
    movies = np.unique(movies)
    movie_idx = np.zeros(movies[-1] + 1, dtype=int)
    for i in range(len(movies)):
        movie_idx[movies[i]] = i
    matrix = np.full((len(users), len(movies)), -1, dtype=np.int32)
    for i in ratings:
        matrix[i[0], movie_idx[i[1]]] = i[2]
    matrix = pd.DataFrame(matrix)
    matrix.replace(-1, np.nan, inplace=True)
    matrix.columns = movies
    return matrix


if __name__ == '__main__':
    file = 'data-202205/train4.txt'
    user_item_matrix = read_file(file)
    user_similar = user_item_matrix.T.corr()
    # print(user_item_matrix)
    # 1. 找出uid用户的相似用户
    similar_users = user_similar[1].drop([1]).dropna()
    # 相似用户筛选规则：正相关的用户
    similar_users = similar_users.where(similar_users > 0).dropna()
    # 2. 从用户1的近邻相似用户中筛选出对物品1有评分记录的近邻用户
    ids = set(user_item_matrix[1].dropna().index) & set(similar_users.index)
    finally_similar_users = similar_users.loc[list(ids)]
    # 3. 结合uid用户与其近邻用户的相似度预测uid用户对iid物品的评分
    numerator = 0  # 评分预测公式的分子部分的值
    denominator = 0  # 评分预测公式的分母部分的值
    for sim_uid, similarity in finally_similar_users.iteritems():
        # 近邻用户的评分数据
        sim_user_rated_movies = user_item_matrix.loc[sim_uid].dropna()
        # 近邻用户对iid物品的评分
        sim_user_rating_for_item = sim_user_rated_movies[1]
        # 计算分子的值
        numerator += similarity * sim_user_rating_for_item
        # 计算分母的值
        denominator += similarity
    # 4 计算预测的评分值
    predict_rating = numerator / denominator
    print("预测出用户<%d>对电影<%d>的评分：%0.2f" % (1, 1, predict_rating))
