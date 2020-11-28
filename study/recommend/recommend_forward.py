# coding:utf-8
# 数据简介：http://files.grouplens.org/datasets/movielens/ml-1m-README.txt
# 数据下载地址：http://files.grouplens.org/datasets/movielens/ml-1m.zip
# import deque
# import next
from collections import deque

import numpy as np
import tensorflow as tf
import time

import study.recommend.readers as readers

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

place_device = "\cpu:0"


def get_data():
    df = readers.read_file("C:\\data\\recommend_data\\ml-1m\\ratings.dat", sep='::')
    rows = len(df)
    print(rows)
    # 打乱数据集
    df = df.iloc[np.random.permutation(rows)].reset_index(drop=True)
    # 分成训练集和验证集
    split_index = int(rows * 0.9)
    # 切割出训练集
    df_train = df[0:split_index]
    # 切割出验证集
    df_test = df[split_index:].reset_index(drop=True)
    return df_train, df_test


def clip(x):
    return np.clip(x, 1.0, 5.0)


def model(user_batch, item_batch, user_num, item_num, dim=5, device="/cpu:0"):
    # with tf.device(device):
    with tf.variable_scope('1si', reuse=True):
        # 全局global
        # 共享操作tf.get_variable
        bias_global = tf.get_variable("bias_global", shape=[])
        # 有user_num个  user的bias
        w_bias_user = tf.get_variable("embd_bias_user", shape=[user_num])
        w_bias_item = tf.get_variable("embd_bias_item", shape=[item_num])
        # 查找出batch个数据
        bias_user = tf.nn.embedding_lookup(w_bias_user, user_batch, name="bias_user")
        bias_item = tf.nn.embedding_lookup(w_bias_item, item_batch, name="bias_item")
        # 对查找出来的user,item进行高斯  dim维度映射
        w_user = tf.get_variable_scope("embd_user", shape=[user_num, dim],
                                       initialize=tf.truncated_normal_initializer(stddev=0.02))
        w_item = tf.get_variable_scope("embd_item", shape=[item_num, dim],
                                       initialize=tf.truncated_normal_initializer(stddev=0.02))
        # 查找出batch个需要更新的数据,
        embd_user = tf.nn.embedding_lookup(w_user, user_batch, name="embdding_user")
        embd_item = tf.nn.embedding_lookup(w_item, item_batch, name="embdding_item")

    # with tf.device(device):
    # 计算公式  r(xi)=u+b(x)+b(i)+q(i)*p(x)
    infer = tf.reduce_sum(tf.multiply(embd_user, embd_item), 1)
    infer = tf.add(infer, bias_global)
    infer = tf.add(infer, bias_user)
    infer = tf.add(infer, bias_item, name="svd_inference")
    # 正则化
    regularizer = tf.add(tf.nn.l2_loss(embd_user), tf.nn.l2_loss(embd_item),
                         name="svd_regularizer")
    return infer, regularizer


def loss(infer, regularizer, rate_batch, learning_rate=0.001, reg=0.1, device="/cpu:0"):
    # with tf.device(device):
    # l2_loss  乘法
    cost_l2 = tf.nn.l2_loss(tf.subtract(infer, rate_batch))
    penalty = tf.constant(reg, dtype=tf.float32, shape=[], name="l2")
    cost = tf.add(cost_l2, tf.multiply(regularizer, penalty))
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)
    return cost, train_op


df_train, df_test = get_data()
samples_per_batch = len(df_train)
iter_train = readers.ShuffleIterator([df_train["user"],
                                      df_train["item"],
                                      df_train["rate"]],
                                     batch_size=batch_size)

iter_test = readers.OneEpochIterator([df_test["user"],
                                      df_test["item"],
                                      df_test["rate"]],
                                     batch_size=-1)

user_batch = tf.placeholder(tf.int32, shape=[None], name="id_user")
item_batch = tf.placeholder(tf.int32, shape=[None], name="id_item")
rate_batch = tf.placeholder(tf.float32, shape=[None])

infer, regularizer = model(user_batch, item_batch, user_num=u_num, item_num=i_num, dim=dims
                           , device=place_device)
_, train_op = loss(infer, regularizer, rate_batch, learning_rate=0.0010, reg=0.05, device=place_device)

saver = tf.train.Saver()
init_op = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init_op)
    print("%s\t%s\t%s\t%s" % ("Epoch", "Train Error", "Val Error", "Elapsed Time"))
    errors = deque(maxlen=samples_per_batch)
    start = time.time()
    for i in range(max_epochs * samples_per_batch):
        users, items, rates = next(iter_train)
        _, pred_batch = sess.run(train_op, infer, feed_dict={user_batch: users, item_batch: items, rate_batch: rates})

        pred_batch = clip(pred_batch)
        errors.append(np.power(pred_batch - rates, 2))
        if i % samples_per_batch == 0:
            train_err = np.sqrt(np.mean(errors))
            test_err2 = np.array([])
            for users, items, rates in iter_test:
                pred_batch = sess.run(infer, feed_dict={user_batch: users, item_batch: items})
                pred_batch = clip(pred_batch)
                test_err2 = np.append(test_err2, np.power(pred_batch - rates, 2))
            end = time.time()
            print("%02d\t%.3f\t\t%.3f secs" % (i // samples_per_batch, train_err, np.sqrt(np.mean(test_err2))))
            start = end
    saver.save(sess, 'C:\\data\\recommend_data\\model')
# print("Number of train samples %d,test samples %d,samples per batch %d" % (len(df_train),
#                                                                            len(df_test), samples_per_batch))
#
# print(df_train["user"].head())
# print(df_test["user"].head())
# print(df_train["item"].head())
# print(df_test["item"].head())
# print(df_train["rate"].head())
# print(df_test["rate"].head())
