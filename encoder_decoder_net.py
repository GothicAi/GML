import tensorflow as tf

from encoder import Encoder
from decoder import Decoder
from facelet import facelet

class EncoderDecoderNet(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()
        with tf.name_scope('feature1CNN'):
            self.fnet1=facelet('feature1', 256)
        with tf.name_scope('feature2CNN'):
            self.fnet2=facelet('feature2', 512)
        with tf.name_scope('feature3CNN'):
            self.fnet3=facelet('feature3', 512)

    def decoder_with_facelet(self,content):
        #encoder doing something
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        content = self.encoder.preprocess(content)
        # encode image
        enc_c, generate_feature = self.encoder.encode(content)
        with tf.name_scope('feature1CNN'):
            feature1 = generate_feature['relu3_1']
            f1t=self.fnet1.combine(feature1,0.01)
        with tf.name_scope('feature2CNN'):
            feature2 = generate_feature['relu4_1']
            f2t=self.fnet2.combine(feature2,0.01)
        with tf.name_scope('feature3CNN'):
            feature3 = generate_feature['relu5_1']
            f3t=self.fnet3.combine(feature3,0.01)

        final_image=self.decoder.decode(f3t,True,f1t,f2t)
        final_image=self.encoder.deprocess(final_image)
        final_image = tf.reverse(final_image, axis=[-1])

        # clip to 0..255
        final_image = tf.clip_by_value(final_image, 0.0, 255.0)
        return final_image

    def decoder_with_deltaV(self,content,f1t,f2t,f3t):
        # encoder doing something
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        content = self.encoder.preprocess(content)
        # encode image
        enc_c, generate_feature = self.encoder.encode(content)
        with tf.name_scope('feature1CNN'):
            feature1 = generate_feature['relu3_1']
            f1t = feature1 + 0.01*f1t
        with tf.name_scope('feature2CNN'):
            feature2 = generate_feature['relu4_1']
            f2t = feature2 + 0.01*f2t
        with tf.name_scope('feature3CNN'):
            feature3 = generate_feature['relu5_1']
            f3t = feature3 + 0.01*f3t

        final_image = self.decoder.decode(f3t, True, f1t, f2t)
        final_image = self.encoder.deprocess(final_image)
        final_image = tf.reverse(final_image, axis=[-1])

        # clip to 0..255
        final_image = tf.clip_by_value(final_image, 0.0, 255.0)

        return final_image

    def decoder_output(self, content):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])
        # preprocess image
        content = self.encoder.preprocess(content)
        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)
        f1 = enc_c_layers['relu3_1']
        f2 = enc_c_layers['relu4_1']
        f3 = enc_c_layers['relu5_1']

        self.encoded_input_layers = enc_c_layers
        self.input_features = enc_c

        # decode target features back to image
        generated_img = self.decoder.decode(f3, False, f1, f2)

        # deprocess image
        generated_img = self.encoder.deprocess(generated_img)

        # switch BGR back to RGB
        generated_img = tf.reverse(generated_img, axis=[-1])

        # clip to 0..255
        generated_img = tf.clip_by_value(generated_img, 0.0, 255.0)
        
        generate_1 = tf.reverse(generated_img, axis=[-1])
        generate_1 = self.encoder.preprocess(generate_1)

        enc_g, enc_g_layers = self.encoder.encode(generate_1)
        self.encoded_generate_layers = enc_g_layers
        self.generate_features = enc_g
        

        return generated_img 
