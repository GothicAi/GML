from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from encoder_decoder_net import EncoderDecoderNet
from utils import get_images, save_images
from utils import get_train_images
import tensorflow.contrib.eager as tfe
import os
import random
import cv2

# tfe.enable_eager_execution()


TRAINING_IMAGE_SHAPE = (256, 256, 3)  # (height, width, color_channels)

EPOCHS = 20
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0

def testall(content_imgs_path, output_image_path, encoder_path, model_save_path,debug=False):
    # get the test image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (1, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='photo')
        # create the style transfer net
        stn = EncoderDecoderNet(encoder_path)
        # pass content and style to the stn, getting the generated_img

        with tf.name_scope("test_group"):
            generated_img = stn.decoder_with_facelet(content)

        sess.run(tf.global_variables_initializer())


        all_photo=os.listdir(content_imgs_path)
        outputs=[]
        print('Now begin to test all model...\n')
        for name in all_photo:
            print("now proceeding:"+name)
            content_img = get_images(content_imgs_path+'/'+name, height=HEIGHT, width=WIDTH)
            img = sess.run(generated_img, feed_dict={content: content_img})
            outputs.append(img[0])
        print('all test done...\n')
    save_images(outputs, all_photo, output_image_path)

#testall('source','log',"vgg19_normalised.npz","")
