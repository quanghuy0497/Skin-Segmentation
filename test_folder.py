import tensorflow as tf
import numpy as np
import utility
import os
import cv2
import random
from train import network

image_path = 'test2/'
MODEL = 'model/'
H, W = 480, 640

def infer(image_paths):
    image_tensor = tf.placeholder(tf.float32, [None, H, W, 3])
    training_flag = tf.placeholder(tf.bool)
    logits = network(image_tensor, training_flag)
    seg_image = tf.argmax(logits, axis=3)

    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, tf.train.latest_checkpoint(MODEL))

        for i, image_path in enumerate(image_paths):
            image = cv2.imread(image_path, 1)
            h, w = image.shape[:2]
            # cv2.imshow('image.jpg', image)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = cv2.resize(image, (W, H))
            image = np.expand_dims(image, axis=0)
            feed_dict = {image_tensor: image,
                         training_flag: True}
            mask = sess.run(seg_image, feed_dict=feed_dict)
            mask = np.squeeze(mask)
            cv2.imwrite(str(image_path) + '.png', mask * 255)
            mask = cv2.imread(str(image_path) + '.png', 1)
            mask = cv2.resize(mask, (w, h))
            cv2.imwrite(str(image_path) + '.png', mask)
            # cv2.imshow('mask.jpg', mask)
            # cv2.destroyAllWindows()  

if __name__ == '__main__':
    # Use this for test
    infer_paths = [image_path + x for x in os.listdir(image_path) if x.endswith('.jpg')]
    print(infer_paths)
    infer(infer_paths)
    print("Done.")