import pandas as pd
import numpy as np


def data_split(data_path, x=0.8, random=False):
    print("读取文件中...")
    f = open(data_path, 'r')
    temp = []
    while True:
        line = f.readline()
        if line == '':
            break
        user, num = line.split('|')
        for i in range(int(num)):
            line = f.readline()
            item, rating = line.split('  ')
            temp.append([int(user), int(item), int(rating)])

    ratings = pd.DataFrame(temp)
    ratings.rename(columns={0: 'userId', 1: 'movieId', 2: 'rating'}, inplace=True)
    print("文件读取完成，正在切分数据集...")
    validation_index = []
    for uid in ratings.groupby("userId").any().index:
        user_rating_data = ratings.where(ratings["userId"] == uid).dropna()
        if random:
            index = list(user_rating_data.index)
            np.random.shuffle(index)
            _index = round(len(user_rating_data) * x)
            validation_index += list(index[_index:])
        else:
            index = round(len(user_rating_data) * x)
            validation_index += list(user_rating_data.index.values[index:])

    validation_set = ratings.loc[validation_index]
    train_set = ratings.drop(validation_index)
    print("完成数据集切分...")
    return train_set, validation_set


if __name__ == '__main__':
    file = 'data-202205/train3.txt'
    train, validation = data_split(file)
    print(train.shape)
    print(validation.shape)
