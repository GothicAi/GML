import tensorflow as tf

from encoder import Encoder
from decoder import Decoder


class EncoderDecoderNet(object):

    def __init__(self, encoder_weights_path):
        self.encoder = Encoder(encoder_weights_path)
        self.decoder = Decoder()

    def decoder_output(self, content):
        # switch RGB to BGR
        content = tf.reverse(content, axis=[-1])

        # preprocess image
        content = self.encoder.preprocess(content)

        # encode image
        enc_c, enc_c_layers = self.encoder.encode(content)

        self.encoded_input_layers = enc_c_layers
        self.input_features = enc_c

        # decode target features back to image
        generated_img = self.decoder.decode(enc_c)

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