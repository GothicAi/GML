from __future__ import print_function

import numpy as np
import tensorflow as tf
from encoder_decoder_net import EncoderDecoderNet
from encoder import Encoder
from utils import get_train_images
from facelet import facelet
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

def train_facelet(content_imgs_path, gt_path, encoder_path,model_save_path):

    # get the traing image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (BATCH_SIZE, HEIGHT, WIDTH, CHANNELS)
    source=np.load(gt_path+"beauty_source.npz")['arr_0']
    target=np.load(gt_path+"beauty_target.npz")['arr_0']
    delta=target-source

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='content')
        gt=tf.placeholder(tf.float32,shape=[1,1703936],name='gt')
        # create the style transfer net
        stn = Encoder(encoder_path)
        # pass content to the stn, getting the feature
        final_feature, generate_feature = stn.encode(content)

        with tf.name_scope('feature1CNN'):
            feature1 = generate_feature['relu3_1']
            fnet1 = facelet('feature1', 256)
            f1t=fnet1.forward(feature1)
            f1tp = tf.layers.flatten(f1t)
        with tf.name_scope('feature2CNN'):
            feature2 = generate_feature['relu4_1']
            fnet2 = facelet('feature2', 512)
            f2t=fnet2.forward(feature2)
            f2tp = tf.layers.flatten(f2t)
        with tf.name_scope('feature3CNN'):
            feature3 = generate_feature['relu5_1']
            fnet3 = facelet('feature3', 512)
            f3t=fnet3.forward(feature3)
            f3tp = tf.layers.flatten(f3t)

        fall=tf.concat([f1tp,f2tp,f3tp],1)
        # compute the loss
        loss = tf.nn.l2_loss(fall-gt)
        global_step = tf.Variable(0, trainable=False)
        learning_rate = tf.train.inverse_time_decay(LEARNING_RATE, global_step, DECAY_STEPS, LR_DECAY_RATE)   
        with tf.name_scope("train_facelet_group"):
            batchnorm_updates = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(batchnorm_updates):
                train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss, global_step=global_step)

        sess.run(tf.global_variables_initializer())
        # saver
        saver = tf.train.Saver(max_to_keep=20)

        ###### Start Training ######
        step = 0
        n_batches = int(len(content_imgs_path) / BATCH_SIZE)
        tf.summary.scalar('loss', loss)
        summary_writer = tf.summary.FileWriter("log", tf.get_default_graph())
        merged = tf.summary.merge_all()
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
                _,loss_num = sess.run([train_op,loss], feed_dict={content: content_batch,gt:delta})

                step += 1
                print(step)
                result = sess.run(merged, feed_dict={content: content_batch, gt: delta})
                summary_writer.add_summary(result, step)

                if step % 500 == 0:
                    saver.save(sess, model_save_path[0], global_step=step, write_meta_graph=False)


            print(loss_num)
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

#            saver.save(sess, model_save_path)
