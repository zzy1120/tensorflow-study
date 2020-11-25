# coding:utf-8
# 数据简介：http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
# 数据下载地址：http://files.grouplens.org/datasets/movielens/ml-1m.zip
# import deque as deque
# import next
import readers

import tensorflow as tf
import numpy as np

import time

# 随机数,seed是随机种子，生产环境是不加的
np.random.seed(24)
# 用户的数量
u_num = 6040
# movies的数量
i_num = 3952

batch_size = 1000
# 隐含因子
dims = 5
# 迭代多少轮
max_epochs = 50


def get_data():
    df = []
    for line in open("C:\\data\\recommend_data\\ml-1m\\ratings.dat", 'r'):
        ls = line.split("::")
        df.append(ls)
    rows=len(df)



if __name__ == '__main__':
    get_data()
