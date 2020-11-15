# coding:utf-8
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import study.mnist_module.mnist_forward as forward
import study.mnist_module.mnist_backward as backward


def restore_model(testPicArr):
    # 重现计算图
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        # 索引最大值就是求得结果
        preValue = tf.argmax(y, 1)
        # 实例化带有滑动平均值的saver
        variable_averages = tf.train.ExponentialMovingAverage(backward.MOVING_AVERAGE_DECAY)
        variable_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variable_to_restore)

        with tf.Session() as sess:
            # 用with结构加载ckpt
            ckpt = tf.train.get_checkpoint_state(backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                # 如果ckpt存在，回复ckpt的参数等当前信息，到会话sess
                saver.restore(sess, ckpt.model_checkpoint_path)
                # 把准备好的未识别图片喂入网络，执行预测操作
                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    img = Image.open(picName)
    # 消除图片锯齿
    reIm = img.resize((28, 28), Image.ANTIALIAS)
    # reIm.convert('L')变成灰度图
    # 图片转换成矩阵的形式
    im_arr = np.array(reIm.convert('L'))
    threshold = 50
    # 对图片进行二值化处理，让图片只有纯白色点，或者纯黑色点，过滤图片中的噪声，留下图片主要特征
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            # 小于阈值的点，视为纯黑的点0
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                # 否则视为纯白色的点255
                im_arr[i][j] = 255
    # 整理形状为1行784列
    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0 / 255.0)
    return img_ready


def application():
    testNum = input("input the number of test pictures: ")
    testNum=int(testNum)
    for i in range(testNum):
        testPic = input("the path of test picture: ")
        testPicArr = pre_pic(testPic)
        # 把治理好的图片喂入神经网络
        preValue = restore_model(testPicArr)
        print("The prediction number is:", preValue)


def main():
    application()


if __name__ == "__main__":
    main()
