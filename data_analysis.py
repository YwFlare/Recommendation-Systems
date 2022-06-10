"""
此文件用于 train.txt 和 test.txt 文件的信息统计
"""
import pandas as pd


def analysis_train(data_path):
    """
    统计训练集信息，包括user数量、item数量、矩阵稀疏度等
    """
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
    print('File size: {}'.format(ratings.shape))
    user = ratings['uid'].value_counts()
    user = user.reset_index()
    user.rename(columns={'index': 'user', 'uid': 'num'}, inplace=True)
    print('User number: {}'.format(user.shape[0]))
    item = ratings['iid'].value_counts()
    item = item.rename('count').reset_index()
    item.rename(columns={'index': 'item', 'iid': 'num'}, inplace=True)
    print('Item number: {}'.format(item.shape[0]))
    print('Matrix vacancy rate: {:.8f}'.format(ratings.shape[0] / (user.shape[0] * item.shape[0])))


def analysis_test(data_path):
    """
    统计验证集信息，包括待遇测的user数量、item数量
    """
    f = open(data_path, 'r')
    users = 0
    items = 0
    while True:
        line = f.readline()
        if line == '':
            break
        user, num = line.split('|')
        users += 1
        items += int(num)
        for i in range(int(num)):
            line = f.readline()
    print('User to test: {}'.format(users))
    print('Item to test: {}'.format(items))


if __name__ == '__main__':
    train = 'data-202205/train.txt'
    test = 'data-202205/test.txt'
    analysis_train(train)
    analysis_test(test)
