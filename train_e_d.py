from __future__ import print_function

import numpy as np
import tensorflow as tf
from encoder_decoder_net import EncoderDecoderNet
from utils import get_train_images
import tensorflow.contrib.eager as tfe
import os
import random
#tfe.enable_eager_execution()


TRAINING_IMAGE_SHAPE = (256, 256, 3) # (height, width, color_channels)

EPOCHS = 20
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0

def transform(image):
    scale_size = 286
    CROP_SIZE = 256

    r = image
    seed = random.randint(0, 2**31 - 1)
    r = tf.image.random_flip_left_right(r, seed=seed)

    # area produces a nice downscaling, but does nearest neighbor for upscaling
    # assume we're going to be doing downscaling here
    r = tf.image.resize_images(r, [scale_size, scale_size], method=tf.image.ResizeMethod.AREA)

    offset = tf.cast(tf.floor(tf.random_uniform([2], 0, scale_size - CROP_SIZE + 1, seed=seed)), dtype=tf.int32)
    if scale_size > CROP_SIZE:
        r = tf.image.crop_to_bounding_box(r, offset[0], offset[1], CROP_SIZE, CROP_SIZE)
    elif scale_size < CROP_SIZE:
        raise Exception("scale size cannot be less than crop size")
    return r

def data_preprocess(imgname):
    image_string = tf.read_file(imgname)
    image_decoded = tf.image.decode_jpeg(image_string)
    image_resized = tf.image.resize_images(image_decoded, [256, 256])
    res = transform(image_resized)
    return res

def train_e_d(content_imgs_path, feature_weight, encoder_path, model_save_path, debug=False, logging_period=100):
    if debug:
        from datetime import datetime
        start_time = datetime.now()

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')

        # create the style transfer net
        stn = EncoderDecoderNet(encoder_path)
        # pass content and style to the stn, getting the generated_img
        generated_img = stn.decoder_output(content)

        # get the target feature maps which is the output of AdaIN
        input_features = stn.input_features
        generate_features = stn.generate_features

        pixel_loss = tf.nn.reduce_mean(tf.abs(generated_img - content))

        feature_loss = tf.nn.reduce_mena(tf.abs(generate_features - input_features))

        # compute the total loss
        loss = pixel_loss + feature_weight * feature_loss
        
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)   
        with tf.name_scope("train_group"):
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(batchnorm_updates):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        # saver
        saver = tf.train.Saver(max_to_keep=20)

        ###### Start Training ######
        step = 0
        n_batches = int(len(content_imgs_path) / BATCH_SIZE)
        
        if debug:
            elapsed_time = datetime.now() - start_time
            start_time = datetime.now()
            print('\nElapsed time for preprocessing before actually train the model: %s' % elapsed_time)
            print('Now begin to train the model...\n')

        #try:
        print('Now begin to train the model...\n')
        for epoch in range(EPOCHS):
            print(epoch)
            np.random.shuffle(content_imgs_path)
            for batch in range(n_batches):
                print(batch)
                    # retrive a batch of content and style images
                content_batch_path = content_imgs_path[batch*BATCH_SIZE:(batch*BATCH_SIZE + BATCH_SIZE)]

                content_batch = get_train_images(content_batch_path, crop_height=HEIGHT, crop_width=WIDTH)
                print('run the training step')
                    # run the training step
                _,loss_log1,loss_log2 = sess.run([train_op,pixel_loss,feature_loss], feed_dict={content: content_batch})

                step += 1
                print(step)

                if step % 1000 == 0:
                    saver.save(sess, model_save_path, global_step=step, write_meta_graph=False)
                print(loss_log1,loss_log2)
            '''
            imgnames = tf.constant(content_imgs_path)
            dataset = tf.data.Dataset.from_tensor_slices(imgnames)
            dataset = dataset.map(data_preprocess)
            dataset = dataset.shuffle(buffer_size=1000).batch(BATCH_SIZE)

            for images in dataset:
                    # run the training step
                _,loss_log=sess.run([train_op,loss], feed_dict={content: images})

                step += 1

                if step % 1000 == 0:
                    saver.save(sess, model_save_path, global_step=step, write_meta_graph=False)
                    print(loss_log)
            '''
        #except Exception as ex:
        #    saver.save(sess, model_save_path, global_step=step)
        #    print('\nSomething wrong happens! Current model is saved to <%s>' % tmp_save_path)
        #    print('Error message: %s' % str(ex))

        ###### Done Training & Save the model ######
        saver.save(sess, model_save_path[0])

#            saver.save(sess, model_save_path)
