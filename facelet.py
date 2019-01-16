import tensorflow as tf
import numpy as np
class facelet(object):
    def __init__(self,name,kernel_num):
        self.weight_vars = []
        self.kernel_num=kernel_num
        with tf.variable_scope(name):
            self.weight_vars.append(self._create_variables(self.kernel_num, self.kernel_num, 3, scope='conv1'))
            self.weight_vars.append(self._create_variables(self.kernel_num, self.kernel_num, 3, scope='conv2'))
            self.weight_vars.append(self._create_variables(self.kernel_num, self.kernel_num, 3, scope='conv3'))

    def _create_variables(self, input_filters, output_filters, kernel_size, scope):
        with tf.variable_scope(scope):
            shape = [kernel_size, kernel_size, input_filters, output_filters]
            kernel = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False), shape=shape,
                                     name='kernel')
            bias = tf.get_variable(initializer=tf.contrib.layers.xavier_initializer(uniform=False),
                                   shape=[output_filters], name='bias')
            return (kernel, bias)

    def _conv2d(self,x, kernel, bias, pad_value=0,use_relu=True):
        # padding image
        x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode="CONSTANT", name='pad', constant_values=pad_value)
        # conv and add bias
        out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
        out = tf.nn.bias_add(out, bias)
        if use_relu:
            out = tf.nn.relu(out)
        return out

    def forward(self, input_feature):
        out=input_feature
        final_layer_idx = len(self.weight_vars) - 1
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]
            if i==0:
                out = self._conv2d(out, kernel, bias, pad_value=1)
            elif i == final_layer_idx:
                out = self._conv2d(out, kernel, bias, pad_value=1, use_relu=False)
            else:
                out = self._conv2d(out, kernel, bias)
        return out
    def combine(self,input_feature,scale=0):
        trans=self.forward(input_feature)
        return scale*trans+input_feature
'''
a=facelet('feature1',128)
f1=tf.placeholder(shape=[None,64,64,128],dtype=tf.float32)
with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    t=sess.run(a.forward(f1),feed_dict={f1:np.zeros(shape=[1,64,64,128])})
    print(t.shape)
'''

