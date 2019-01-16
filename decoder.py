import tensorflow as tf
import numpy as np
slim = tf.contrib.slim
class Decoder(object):

    def __init__(self):
        self.weight_vars = []
        with tf.variable_scope('decoder'):
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv5_1'))
            
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv4_4'))
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv4_3'))
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv4_2'))#here add feature 2
            self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))#here add feature 1
            self.weight_vars.append(self._create_variables(256, 128, 3, scope='conv3_1'))

            self.weight_vars.append(self._create_variables(128, 128, 3, scope='conv2_2'))
            self.weight_vars.append(self._create_variables(128,  64, 3, scope='conv2_1'))

            self.weight_vars.append(self._create_variables( 64,  64, 3, scope='conv1_2'))
            self.weight_vars.append(self._create_variables( 64,   3, 3, scope='conv1_1'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape  = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=shape, name='kernel')
            bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=[output_filters], name='bias')
            return (kernel, bias)

    def _batchnorm(self,input,is_training):
        with tf.variable_scope("batchnorm"):
            # this block looks like it has 3 inputs on the graph unless we do this
            if is_training:
                normalized = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=0.001,
                                                          is_training=True, fused=True, scope='bn')
            else:
                normalized = slim.batch_norm(input, decay=0.99, center=True, scale=True, epsilon=0.001,
                                                          is_training=False, fused=True, scope='bn')
            return normalized

    def decode(self, image,using_facelet=False,f1=None,f2=None,is_training=False):
        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = (0, 4, 8,10)
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            #print(out.shape)
            kernel, bias = self.weight_vars[i]
            if using_facelet:
                if i == 7:  # f1 index
                    out = self._conv2d(out, kernel, bias,is_training) + f1
                elif i == 3:  # f2 index
                    out = self._conv2d(out, kernel, bias,is_training) + f2
                elif i == final_layer_idx:
                    out = self._conv2d(out, kernel, bias,is_training,use_relu=False)
                else:
                    out = self._conv2d(out, kernel, bias,is_training)
            else:
                if i == final_layer_idx:
                    out = self._conv2d(out, kernel, bias,is_training, use_relu=False)
                else:
                    out = self._conv2d(out, kernel, bias,is_training)
            
            if i in upsample_indices:
                out = upsample(out)

        return out
    def _conv2d(self,x, kernel, bias, is_training,use_relu=True):
        name=kernel.name.split('/')[-2]
        with tf.variable_scope(name):
            # padding image with reflection mode
            x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

            # conv and add bias
            out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
            out = tf.nn.bias_add(out, bias)
            #out= self._batchnorm(out,is_training)
            if use_relu:
                out = tf.nn.relu(out)

            return out


def upsample(x, scale=2):
    height = tf.shape(x)[1] * scale
    width  = tf.shape(x)[2] * scale
    output = tf.image.resize_images(x, [height, width], 
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return output
'''
a=Decoder()
f3=tf.placeholder(shape=[1,16,16,512],dtype=tf.float32)

update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
with tf.control_dependencies(update_ops):
    xm=a.decode(f3)

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    sess.run(tf.local_variables_initializer())
    t=sess.run(xm,feed_dict={f3:np.zeros(shape=[1,16,16,512])})
    print(t.shape)
'''

'''
    def decode(self, image):
        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = (0, 4, 6)

        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            
            kernel, bias = self.weight_vars[i]
            if i in upsample_indices:
                stride = 2
            else:
                stride = 1
                
            if i == final_layer_idx:
                out = deconv(out, bias, stride)
            else:
                out = deconv(out, bias, stride)
        return out
    
def deconv(batch_input, out_channels, stride):
    with tf.variable_scope("deconv"):
        batch, in_height, in_width, in_channels = [int(d) for d in batch_input.get_shape()]
        filter = tf.get_variable("filter", [4, 4, out_channels, in_channels], dtype=tf.float32, initializer=tf.random_normal_initializer(0, 0.02))
        conv = tf.nn.conv2d_transpose(batch_input, filter, [batch, in_height * stride, in_width * stride, out_channels], [1, stride, stride, 1], padding="SAME")
        return conv
'''