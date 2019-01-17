from __future__ import print_function

import numpy as np
import os
import tensorflow as tf
from encoder_decoder_net import EncoderDecoderNet
from utils import get_images, save_images
from utils import get_train_images
#import tensorflow.contrib.eager as tfe
import os
import random
import cv2
from tensorflow.python.framework import graph_util

# tfe.enable_eager_execution()
content_image_path = 'input_img'
output_image_path = 'outputs'
encoder_path = 'vgg19_normalised.npz'
#model_save_path = 'Aug_L1_models'
pretrained_path_encoder_decoder = 'pretrained_encoder_decoder_model'
pretrained_path_facelet = 'pretrained_facelet_model_hair'

TRAINING_IMAGE_SHAPE = (256, 256, 3)  # (height, width, color_channels)

EPOCHS = 20
BATCH_SIZE = 24
LEARNING_RATE = 1e-4
LR_DECAY_RATE = 5e-5
DECAY_STEPS = 1.0

def testall(content_imgs_path, output_image_path, encoder_path, pretrained_path_encoder_decoder, pretrained_path_facelet, model_save_path=None):
    # get the test image shape
    HEIGHT, WIDTH, CHANNELS = TRAINING_IMAGE_SHAPE
    INPUT_SHAPE = (1, HEIGHT, WIDTH, CHANNELS)

    # create the graph
    with tf.Graph().as_default(), tf.Session() as sess:
        source = np.load("senior_source.npz")['arr_0']
        target = np.load("senior_target.npz")['arr_0']
        deltav = target - source
        v1 = deltav[0, 0:1 * 64 * 64 * 256].reshape([1, 64, 64, 256])
        v2 = deltav[0, 1 * 64 * 64 * 256:1 * 64 * 64 * 256 + 32 * 32 * 512].reshape([1, 32, 32, 512])
        v3 = deltav[0, 1 * 64 * 64 * 256 + 32 * 32 * 512:].reshape([1, 16, 16, 512])

        tv1 = tf.constant(v1)
        tv2 = tf.constant(v2)
        tv3 = tf.constant(v3)
        content = tf.placeholder(tf.float32, shape=INPUT_SHAPE, name='photo')
        # create the style transfer net
        stn = EncoderDecoderNet(encoder_path)
        # pass content and style to the stn, getting the generated_img

        with tf.name_scope("test_group"):
            #generated_img=stn.decoder_output(content,tv1,tv2,tv3)
	    #generated_img=stn.decoder_with_deltaV(content,tv1,tv2,tv3)
	    generated_img = stn.decoder_with_facelet(content)
        exclusions = ['encoder', '/encoder', 'feature1', '/feature1', 'feature2', '/feature2','feature3', '/feature3']
        variables_to_restore = []
        # for var in tf.contrib.framework.get_model_variables():
        for var in tf.contrib.slim.get_variables_to_restore():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
            # print(variables_to_restore)
        saver_ed = tf.train.Saver(variables_to_restore, max_to_keep=20)

        #print(pretrained_path_facelet)
        #print(pretrained_path_encoder_decoder)
        exclusions = ['encoder', '/encoder', 'decoder', '/decoder']
        variables_to_restore = []
        # for var in tf.contrib.framework.get_model_variables():
        for var in tf.contrib.slim.get_variables_to_restore():
            excluded = False
            for exclusion in exclusions:
                if var.op.name.startswith(exclusion):
                    excluded = True
                    break
            if not excluded:
                variables_to_restore.append(var)
        # print(variables_to_restore)
        saver_facelet = tf.train.Saver(variables_to_restore, max_to_keep=20)

        saver_all = tf.train.Saver(tf.contrib.slim.get_variables_to_restore(), max_to_keep=50)

        sess.run(tf.global_variables_initializer())


        if model_save_path is not None:
            print("loading model from checkpoint")
            checkpoint = tf.train.latest_checkpoint(model_save_path)
            saver_all.restore(sess, checkpoint)
        else:
            print("\n\tLoading pre-trained encoder-decoder model ...")
            checkpoint_dir = os.path.join(pretrained_path_encoder_decoder)
            checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoints and checkpoints.model_checkpoint_path:
                checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
                saver_ed.restore(sess, os.path.join(checkpoint_dir, checkpoints_name))
            print("\n\tLoading pre-trained facelet model ...")
            checkpoint_dir = os.path.join(pretrained_path_facelet)
            checkpoints = tf.train.get_checkpoint_state(checkpoint_dir)
            if checkpoints and checkpoints.model_checkpoint_path:
                checkpoints_name = os.path.basename(checkpoints.model_checkpoint_path)
                saver_facelet.restore(sess, os.path.join(checkpoint_dir, checkpoints_name))
        '''
        output_graph_def = graph_util.convert_variables_to_constants(sess,sess.graph_def,["test_group/clip_by_value"])
        filename = os.path.join(output_image_path,"faceletbank_beauty_inference.pb")
        with tf.gfile.GFile(filename, "wb") as f:
            f.write(output_graph_def.SerializeToString())
        print("%d ops in the final graph." % len(output_graph_def.node))
        '''
        #summary_writer = tf.summary.FileWriter("log", tf.get_default_graph())
        #merged = tf.summary.merge_all()

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
    #result = sess.run(merged, feed_dict={content: content_batch})
    #summary_writer.add_summary(result, step)

testall(content_image_path, output_image_path, encoder_path, pretrained_path_encoder_decoder, pretrained_path_facelet)
