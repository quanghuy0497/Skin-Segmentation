import tensorflow as tf
import numpy as np
import utility
import os
import cv2
import random
from train import network

MODEL = 'model/'
H, W = 480, 640
image_path = "test4.jpg"

if __name__ == '__main__':
    image_tensor = tf.placeholder(tf.float32, [None, H, W, 3])
    training_flag = tf.placeholder(tf.bool)
    logits = network(image_tensor, training_flag)
    seg_image = tf.argmax(logits, axis=3)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL))
        img = cv2.imread(image_path)
        h, w = img.shape[:2]
        cv2.imshow('image.jpg', img)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (W, H))
        img = np.expand_dims(img, axis=0)
        feed_dict = {image_tensor: img,
                    training_flag: True}
        mask = sess.run(seg_image, feed_dict=feed_dict)
        mask = np.squeeze(mask)
        cv2.imwrite(str(image_path) + '.png', mask * 255)
        mask = cv2.imread(str(image_path) + '.png', 1)
        mask = cv2.resize(mask, (w, h))
        cv2.imwrite(str(image_path) + '.png', mask)
        cv2.imshow('mask.jpg', mask)
        cv2.waitKey(0)