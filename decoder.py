import tensorflow as tf


class Decoder(object):

    def __init__(self):
        self.weight_vars = []

        with tf.variable_scope('decoder'):
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv5_1'))
            
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv4_4'))
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv4_3'))
            self.weight_vars.append(self._create_variables(512, 512, 3, scope='conv4_2'))
            self.weight_vars.append(self._create_variables(512, 256, 3, scope='conv4_1'))

            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_4'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_3'))
            self.weight_vars.append(self._create_variables(256, 256, 3, scope='conv3_2'))
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
    

    def decode(self, image):
        # upsampling after 'conv4_1', 'conv3_1', 'conv2_1'
        upsample_indices = (0, 4, 6)
        final_layer_idx  = len(self.weight_vars) - 1

        out = image
        for i in range(len(self.weight_vars)):
            kernel, bias = self.weight_vars[i]

            if i == final_layer_idx:
                out = conv2d(out, kernel, bias, use_relu=False)
            else:
                out = conv2d(out, kernel, bias)
            
            if i in upsample_indices:
                out = upsample(out)

        return out

def conv2d(x, kernel, bias, use_relu=True):
    # padding image with reflection mode
    x_padded = tf.pad(x, [[0, 0], [1, 1], [1, 1], [0, 0]], mode='REFLECT')

    # conv and add bias
    out = tf.nn.conv2d(x_padded, kernel, strides=[1, 1, 1, 1], padding='VALID')
    out = tf.nn.bias_add(out, bias)

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