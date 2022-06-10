"""
此文件用于划分数据集和验证集
"""
import pandas as pd
import numpy as np
from d2l import torch as d2l


def data_split(data_path, x=0.9, random=False):
    """
    切分数据集和验证集，默认数据集和验证集比例为 9:1
    """
    print("Split data ...")
    timer = d2l.Timer()
    timer.start()
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
    ratings.rename(columns={0: 'uid', 1: 'iid', 2: 'rating'}, inplace=True)
    validation_index = []
    for uid in ratings.groupby("uid").any().index:
        user_rating_data = ratings.where(ratings["uid"] == uid).dropna()
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
    timer.stop()
    print('Takes :{} sec'.format(timer.sum()))
    return train_set, validation_set


if __name__ == '__main__':
    file = 'data-202205/train3.txt'
    train, validation = data_split(file)
    print(train.shape)
    print(validation.shape)
