
# coding: utf-8

# In[1]:

import tensorflow as tf

print(1)
import numpy as np
import tensorflow.contrib.slim.nets as nets


slim = tf.contrib.slim

image=tf.placeholder(shape=[1,256,256,3],dtype=tf.float32,name='start')

g = tf.Graph()

with tf.variable_scope('feature1') as sc:
    feature1 = slim.conv2d(image, 64, [3, 3], stride=1, padding='SAME', scope='conv1_1',activation_fn=tf.nn.relu)
    feature1 = slim.conv2d(feature1, 64, [3, 3], stride=1, padding='SAME', scope='conv1_2',activation_fn=tf.nn.relu)
    feature1 = slim.max_pool2d(feature1, [2, 2], scope='pool1')
    feature1 = slim.conv2d(feature1, 128, [3, 3], stride=1, padding='SAME', scope='conv2_1',activation_fn=tf.nn.relu)
    feature1= slim.conv2d(feature1, 128, [3, 3], stride=1, padding='SAME', scope='conv2_2',activation_fn=tf.nn.relu)
    feature1 = slim.max_pool2d(feature1, [2, 2], scope='pool2')
    feature1 = slim.conv2d(feature1, 256, [3, 3], stride=1, padding='SAME', scope='conv3_1',activation_fn=tf.nn.relu)
with tf.variable_scope('feature2') as sc:
    feature2 = slim.conv2d(feature1, 256, [3, 3], stride=1, padding='SAME', scope='conv3_2',activation_fn=tf.nn.relu)
    feature2 = slim.conv2d(feature2, 256, [3, 3], stride=1, padding='SAME', scope='conv3_3',activation_fn=tf.nn.relu)
    feature2 = slim.conv2d(feature2, 256, [3, 3], stride=1, padding='SAME', scope='conv3_4',activation_fn=tf.nn.relu)
    feature2 = slim.max_pool2d(feature2, [2, 2], scope='pool3')
    feature2 = slim.conv2d(feature2, 512, [3, 3], stride=1, padding='SAME', scope='conv4_1',activation_fn=tf.nn.relu)
with tf.variable_scope('feature3') as sc:
    feature3 = slim.conv2d(feature2, 512, [3, 3], stride=1, padding='SAME', scope='conv4_2', activation_fn=tf.nn.relu)
    feature3 = slim.conv2d(feature3, 512, [3, 3], stride=1, padding='SAME', scope='conv4_3', activation_fn=tf.nn.relu)
    feature3 = slim.conv2d(feature3, 512, [3, 3], stride=1, padding='SAME', scope='conv4_4', activation_fn=tf.nn.relu)
    feature3 = slim.max_pool2d(feature3, [2, 2], scope='pool4')
    feature3 = slim.conv2d(feature3, 512, [3, 3], stride=1, padding='SAME', scope='conv5_1', activation_fn=tf.nn.relu)

with tf.variable_scope('f1t') as sc:
    f1t=slim.conv2d(
        tf.pad(feature1,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT", name='pad', constant_values=1),
        256,
        [3,3],
        stride=1,
        padding='VALID',
        scope='conv1',
        activation_fn=tf.nn.relu
    )
    f1t=slim.conv2d(f1t, 256, [3, 3], stride=1, padding='SAME', scope='conv2', activation_fn=tf.nn.relu)
    f1t=slim.conv2d(
        tf.pad(f1t,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT", name='pad', constant_values=1),
        256,
        [3,3],
        stride=1,
        padding='VALID',
        scope='conv3',
        activation_fn=tf.nn.relu
    )
    f1t=f1t+feature1
with tf.variable_scope('f2t') as sc:
    f2t=slim.conv2d(
        tf.pad(feature2,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT", name='pad', constant_values=1),
        512,
        [3,3],
        stride=1,
        padding='VALID',
        scope='conv1',
        activation_fn=tf.nn.relu
    )
    f2t=slim.conv2d(f2t, 512, [3, 3], stride=1, padding='SAME', scope='conv2', activation_fn=tf.nn.relu)
    f2t=slim.conv2d(
        tf.pad(f2t,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT", name='pad', constant_values=1),
        512,
        [3,3],
        stride=1,
        padding='VALID',
        scope='conv3',
        activation_fn=tf.nn.relu
    )
    f2t=f2t+feature2
with tf.variable_scope('f3t') as sc:
    f3t=slim.conv2d(
        tf.pad(feature3,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT", name='pad', constant_values=1),
        512,
        [3,3],
        stride=1,
        padding='VALID',
        scope='conv1',
        activation_fn=tf.nn.relu
    )
    f3t=slim.conv2d(f3t, 512, [3, 3], stride=1, padding='SAME', scope='conv2', activation_fn=tf.nn.relu)
    f3t=slim.conv2d(
        tf.pad(f3t,[[0,0],[1,1],[1,1],[0,0]],mode="CONSTANT", name='pad', constant_values=1),
        512,
        [3,3],
        stride=1,
        padding='VALID',
        scope='conv3',
        activation_fn=tf.nn.relu
    )
    f3t=f3t+feature3





def Bilinear(net,scale=2):
    with tf.variable_scope('resize'):
        return tf.image.resize_images(net, [tf.shape(net)[1] * 2, tf.shape(net)[1] * 2],method=tf.image.ResizeMethod.BILINEAR)

def _Upsample(input_tensor,n_output_filters):
    return slim.conv2d(input_tensor, n_output_filters, [3, 3], stride=1, padding='SAME', scope='upsample', normalizer_fn=slim.batch_norm,activation_fn=None)

def _TransitionUp(input_tensor,n_output_filters):
    return slim.conv2d_transpose(input_tensor,n_output_filters,[4,4],stride=2,padding='SAME',scope='transitionup',normalizer_fn=slim.batch_norm,activation_fn=None)

def _PollingBlock(input_tensor, n_convs, n_output_filters):
    return slim.repeat(input_tensor,n_convs,slim.conv2d,n_output_filters,[3,3],stride=1,padding='SAME',scope='pollingblock',normalizer_fn=slim.batch_norm,activation_fn=tf.nn.relu)


with tf.variable_scope('recon5') as sc:
    net = _PollingBlock(f3t,3,512);
    net=Bilinear(net)

with tf.variable_scope('upool4') as sc:
    net=_Upsample(net,512)


with tf.variable_scope('recon4') as sc:
    net=_PollingBlock(net+f2t,3,512)
    net = Bilinear(net)

with tf.variable_scope('upool3') as sc:
    net = _Upsample(net, 256)

with tf.variable_scope('recon3') as sc:
    net=_PollingBlock(net+f1t,3,256)
    net = Bilinear(net)

with tf.variable_scope('upool2') as sc:
    net = _Upsample(net, 128)

with tf.variable_scope('recon2') as sc:
    net=_PollingBlock(net,2,128)
    net = Bilinear(net)

with tf.variable_scope('upool1') as sc:
    net = _Upsample(net, 64)

with tf.variable_scope('recon1') as sc:
    net=_PollingBlock(net,1,64)

with tf.variable_scope('recon0') as sc:
    net=slim.conv2d(net, 3, [3, 3], stride=1, padding='SAME', scope='conv', activation_fn=None)

out = tf.Variable(net, name='final')

#slim.losses.softmax_cross_entropy(predictions, label)
#total_loss = slim.losses.get_total_loss()
# create_train_op ensures that each time we ask for the loss, the update_ops
# are run and the gradients being computed are applied too.
#train_op = slim.learning.create_train_op(total_loss, optimizer)
print('1')
init_op=tf.global_variables_initializer()
saver = tf.train.Saver(max_to_keep=4)
#创建session
with tf.Session() as sess:
    # 变量初始化
    sess.run(init_op,feed_dict={image:np.zeros([1,256,256,3],dtype=np.float32)})
    #print('1')
    a,b,c=sess.run([f1t,f2t,net],feed_dict={image:np.zeros([1,256,256,3],dtype=np.float32)})
    #print(a.shape)
    print(b.shape)
    print(c.shape)
    saver.save(sess,save_path="model/")
    tf.train.write_graph(sess.graph_def, 'model/', 'model.pb')
    tf.summary.FileWriter("./log", tf.get_default_graph())


