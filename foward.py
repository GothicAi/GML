
# coding: utf-8

# In[1]:

import tensorflow as tf
import numpy as np
import tensorflow.contrib.slim.nets as nets

from tensorflow.examples.tutorials.mnist import input_data

slim = tf.contrib.slim
vgg = nets.vgg

# 超参数
learning_rate = 0.5
epochs = 12
batch_size = 10

image=tf.placeholder(shape=[1,256,256,3],dtype=tf.float32)

g = tf.Graph()

# Create the model and specify the losses...
optimizer = tf.train.GradientDescentOptimizer(learning_rate)

#with tf.variable_scope('feature1') as sc:
feature1 = slim.conv2d(image, 64, [3, 3], stride=1, padding='SAME', scope='conv1_1',activation_fn=tf.nn.relu)
feature1 = slim.conv2d(feature1, 64, [3, 3], stride=1, padding='SAME', scope='conv1_2',activation_fn=tf.nn.relu)
feature1 = slim.max_pool2d(feature1, [2, 2], scope='pool1')
feature1 = slim.conv2d(feature1, 128, [3, 3], stride=1, padding='SAME', scope='conv2_1',activation_fn=tf.nn.relu)
feature1= slim.conv2d(feature1, 128, [3, 3], stride=1, padding='SAME', scope='conv2_2',activation_fn=tf.nn.relu)
feature1 = slim.max_pool2d(feature1, [2, 2], scope='pool2')
feature1 = slim.conv2d(feature1, 256, [3, 3], stride=1, padding='SAME', scope='conv3_1',activation_fn=tf.nn.relu)
#with tf.variable_scope('feature2') as sc:
feature2 = slim.conv2d(feature1, 256, [3, 3], stride=1, padding='SAME', scope='conv3_2',activation_fn=tf.nn.relu)
feature2 = slim.conv2d(feature2, 256, [3, 3], stride=1, padding='SAME', scope='conv3_3',activation_fn=tf.nn.relu)
feature2 = slim.conv2d(feature2, 256, [3, 3], stride=1, padding='SAME', scope='conv3_4',activation_fn=tf.nn.relu)
feature2 = slim.max_pool2d(feature2, [2, 2], scope='pool3')
feature2 = slim.conv2d(feature2, 512, [3, 3], stride=1, padding='SAME', scope='conv4_1',activation_fn=tf.nn.relu)
#with tf.variable_scope('feature3') as sc:
feature3 = slim.conv2d(feature2, 512, [3, 3], stride=1, padding='SAME', scope='conv4_2', activation_fn=tf.nn.relu)
feature3 = slim.conv2d(feature3, 512, [3, 3], stride=1, padding='SAME', scope='conv4_3', activation_fn=tf.nn.relu)
feature3 = slim.conv2d(feature3, 512, [3, 3], stride=1, padding='SAME', scope='conv4_4', activation_fn=tf.nn.relu)
feature3 = slim.max_pool2d(feature3, [2, 2], scope='pool4')
feature3 = slim.conv2d(feature3, 512, [3, 3], stride=1, padding='SAME', scope='conv5_1', activation_fn=tf.nn.relu)

#with tf.variable_scope('f1t') as sc:
f1t=slim.conv2d(
    tf.padding(feature1,[0,1,1,0],mode="CONSTANT", name='pad', constant_values=1),
    256,
    [3,3],
    stride=1,
    padding='VALID',
    scope='conv1',
    activation_fn=tf.nn.relu
)
f1t=slim.conv2d(f1t, 256, [3, 3], stride=1, padding='SAME', scope='conv2', activation_fn=tf.nn.relu)
f1t=slim.conv2d(
    tf.padding(f1t,[0,1,1,0],mode="CONSTANT", name='pad', constant_values=1),
    256,
    [3,3],
    stride=1,
    padding='VALID',
    scope='conv3',
    activation_fn=tf.nn.relu
)
f1t=f1t+feature1
#with tf.variable_scope('f2t') as sc:
f2t=slim.conv2d(
    tf.padding(feature2,[0,1,1,0],mode="CONSTANT", name='pad', constant_values=1),
    512,
    [3,3],
    stride=1,
    padding='VALID',
    scope='conv1',
    activation_fn=tf.nn.relu
)
f2t=slim.conv2d(f2t, 512, [3, 3], stride=1, padding='SAME', scope='conv2', activation_fn=tf.nn.relu)
f2t=slim.conv2d(
    tf.padding(f2t,[0,1,1,0],mode="CONSTANT", name='pad', constant_values=1),
    512,
    [3,3],
    stride=1,
    padding='VALID',
    scope='conv3',
    activation_fn=tf.nn.relu
)
f2t=f2t+feature2
#with tf.variable_scope('f3t') as sc:
f3t=slim.conv2d(
    tf.padding(feature3,[0,1,1,0],mode="CONSTANT", name='pad', constant_values=1),
    512,
    [3,3],
    stride=1,
    padding='VALID',
    scope='conv1',
    activation_fn=tf.nn.relu
)
f3t=slim.conv2d(f3t, 512, [3, 3], stride=1, padding='SAME', scope='conv2', activation_fn=tf.nn.relu)
f3t=slim.conv2d(
    tf.padding(f3t,[0,1,1,0],mode="CONSTANT", name='pad', constant_values=1),
    512,
    [3,3],
    stride=1,
    padding='VALID',
    scope='conv3',
    activation_fn=tf.nn.relu
)
f3t=f3t+feature3


#slim.losses.softmax_cross_entropy(predictions, label)
#total_loss = slim.losses.get_total_loss()

# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are applied too.
#train_op = slim.learning.create_train_op(total_loss, optimizer)

init_op=tf.initialize_all_variables()
saver = tf.train.Saver(max_to_keep=4)
#创建session
with tf.Session() as sess:
    # 变量初始化
    sess.run(init_op)
    sess.run(f1t,feed_dict={image:np.zeros([1,256,256,3])})
    
