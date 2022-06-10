"""
此文件用于获取 itemAttribute 中的 Attribute 并对其进行处理
"""
import pandas as pd
from d2l import torch as d2l


def read_att(attribute_path):
    """
    读取 itemAttribute.txt 文件，将item与Attribute以键值对的形式保存到字典
    """
    f = open(attribute_path, 'r')
    pre_att = {}
    while True:
        line = f.readline()
        if line == '':
            break
        item, att1, att2 = line.split('|')
        pre_att[item] = (att1, att2[:-1])
    return pre_att


def get_att(data_path, att, index):
    """
    获取 user 与 Attribute 的对应关系，若有多个则取其均值
    """
    print("Attribute processing ...")
    timer = d2l.Timer()
    timer.start()
    f = open(data_path, 'r')
    attribute = pd.DataFrame(columns=['uid', 'iid', 'rating'], dtype=float)
    while True:
        line = f.readline()
        if line == '':
            break
        user, num = line.split('|')
        group = pd.DataFrame(columns=['iid', 'rating'], dtype=float)
        for i in range(int(num)):
            line = f.readline()
            item, rating = line.split('  ')
            if item in att.keys() and att[item][index] != 'None':
                temp = pd.DataFrame([[int(att[item][index]), int(rating)]], columns=['iid', 'rating'])
                group = group.append(temp)
        if group.empty:
            continue
        group = group.groupby('iid').mean().reset_index()
        group['uid'] = int(user)
        attribute = attribute.append(group[['uid', 'iid', 'rating']], ignore_index=True)
    timer.stop()
    print('Takes :{} sec'.format(timer.sum()))
    return attribute


if __name__ == '__main__':
    file = 'data-202205/train6.txt'
    path = 'data-202205/itemAttribute1.txt'
    data_attribute = read_att(path)
    get_att(file, data_attribute, 0)
