# coding:utf-8
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import study.mnist_module.mnist_forward as forward
import study.mnist_module.mnist_generateds as mnist_generateds
import os

BATCH_SIZE = 200
LEARNING_RATE_BASE = 0.1
LEARNING_RATE_DECAY = 0.99
REGULARIZER = 0.0001
STEPS = 50000
MOVING_AVERAGE_DECAY = 0.99
MODEL_SAVE_PATH = "c:/data/models/"
MODEL_NAME = "mnist_model"
# 训练的总样本数是60000
train_num_examples = 60000


def backward():
    x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
    y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
    y = forward.forward(x, REGULARIZER)
    # 赋初值0，不可训练
    global_step = tf.Variable(0, trainable=False)

    # 调用包含正则化的损失函数loss
    ce = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=y, labels=tf.argmax(y_, 1))
    cem = tf.reduce_mean(ce)
    loss = cem + tf.add_n(tf.get_collection('losses'))

    # 定义指数衰减学习率
    learning_rate = tf.train.exponential_decay(
        LEARNING_RATE_BASE,
        global_step,
        train_num_examples / BATCH_SIZE,
        LEARNING_RATE_DECAY,
        staircase=True)
    # 定义训练过程
    train_op = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

    # 定义滑动平均
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    ema_op = ema.apply(tf.trainable_variables())
    with tf.control_dependencies([train_op, ema_op]):
        train_op = tf.no_op(name='train')

    # 实例化saver
    saver = tf.train.Saver()

    # 新增 批量获取图片，标签
    img_batch, label_batch = mnist_generateds.get_tfrecord(BATCH_SIZE, isTrain=True)

    with tf.Session() as sess:
        # 初始化所有变量
        init_op = tf.global_variables_initializer()
        sess.run(init_op)

        # 断点续训
        ckpt = tf.train.get_checkpoint_state(MODEL_SAVE_PATH)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)

        # 增加线程协调器代码
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=sess, coord=coord)

        for i in range(STEPS):
            # 旧代码 xs, ys = mnist.train.next_batch(BATCH_SIZE)
            # 新增代码
            # xs, ys = sess.run(img_batch, label_batch)
            xs, ys = sess.run([img_batch, label_batch])

            _, loss_value, step = sess.run([train_op, loss, global_step], feed_dict={x: xs, y_: ys})
            if i % 1000 == 0:
                print("After %d training step(s),loss on training batch is %g." % (step, loss_value))
                saver.save(sess, os.path.join(MODEL_SAVE_PATH, MODEL_NAME), global_step=global_step)
        # 线程停止
        coord.request_stop()
        coord.join(threads)


def main():
    mnist = input_data.read_data_sets("c:/data/mnist/", one_hot=True)
    backward(mnist)


if __name__ == '__main__':
    main()
