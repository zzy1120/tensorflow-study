# coding:utf-8
import tensorflow as tf
import numpy as np
import PIL.Image as Image
import os

image_train_path = 'c:/data/mnist_data_jpg/mnist_train_jpg_60000'
label_train_path = 'c:/data/mnist_data_jpg/mnist_train_jpg_60000.txt'
tfRecord_train = 'c:/data/mnist_train.tfrecords'
image_test_path = 'c:/data/mnist_data_jpg/mnist_test_jpg_10000'
lable_test_path = 'c:/data/mnist_data_jpg/mnist_test_jpg_10000.txt'
tfRecord_test = 'c:/data/mnist_test.tfrecords'
data_path = 'c:/data'
resize_height = 28
resize_width = 28


def writer_tfRecord(tfRecordName, image_path, label_path):
    writer = tf.python_io.TFRecordWriter(tfRecordName)
    num_pic = 0
    f = open(label_path, 'r')
    contents = f.readlines()
    f.close()
    for content in contents:
        value = content.split()
        img_path = image_path + value[0]
        img=Image.open(image_path)
        img_raw=img.tobytes()
        labels=[0]*10
        labels[int[value[1]]]=1

def generate_tfRecord():
    isExists = os.path.exists(data_path)
    if not isExists:
        os.makedirs(data_path)
        print("The directory was created successfully")
    else:
        print("directory is exists")
    writer_tfRecord(tfRecord_train, image_train_path, label_train_path)
    writer_tfRecord(tfRecord_test, image_test_path, lable_test_path)


def main():
    generate_tfRecord()


if __name__ == '__main__':
    main()
