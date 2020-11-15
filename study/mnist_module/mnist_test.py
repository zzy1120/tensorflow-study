# coding:utf-8
import time
import tensorflow as tf
import tensorflow.examples.tutorials.mnist.input_data as input_data
import study.mnist_module.mnist_forward as forward
import study.mnist_module.mnist_backward as backward

TEST_INTERVAL_SECS = 5


def test(mnist):
    # 复现计算图
    with tf.Graph().as_default() as g:
        # x占位
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y_ = tf.placeholder(tf.float32, [None, forward.OUTPUT_NODE])
        y = forward.forward(x, None)

        # 实例化带滑动平均的saver对象
        ema = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        ema_restore = ema.variables_to_restore()
        saver = tf.train.Saver(ema_restore)
        # 计算准确率
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        while True:
            with tf.Session() as sess:
                # 把滑动平均值赋给各个参数
                ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
                # 先判断是否已经有模型
                if ckpt and ckpt.model_checkpoint_path:
                    # 恢复模型到当前会话
                    saver.restore(sess, ckpt.model_checkpoint_path)
                    # 恢复global_step值
                    global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
                    # 执行准确率计算
                    accuracy_score = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                    print("After %s trainning step(s),test accuracy = %g" % (global_step, accuracy_score))
                else:
                    print("No checkpoint file found")
                    return
            time.sleep(TEST_INTERVAL_SECS)


def main():
    mnist = input_data.read_data_sets("c:/data/mnist/", one_hot=True)
    test(mnist)


if __name__ == '__main__':
    main()
