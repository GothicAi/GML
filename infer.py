import tensorflow as tf

from encoder_decoder_net import EncoderDecoderNet
from utils import get_images, save_images


def stylize(contents_path, output_dir, encoder_path, model_path, resize_height=None, resize_width=None, suffix=None):

    if isinstance(contents_path, str):
        contents_path = [contents_path]

    with tf.Graph().as_default(), tf.Session() as sess:
        # build the dataflow graph
        content = tf.placeholder(tf.float32, shape=(1, None, None, 3), name='content')

        stn = EncoderDecoderNet(encoder_path)

        output_image = stn.decoder_output(content)

        sess.run(tf.global_variables_initializer())

        # restore the trained model and run the style transferring
        saver = tf.train.Saver()
        saver.restore(sess, model_path[0])

        outputs = []
        for content_path in contents_path:

            content_img = get_images(content_path, height=resize_height, width=resize_width)

            result = sess.run(output_image, feed_dict={content: content_img})

            outputs.append(result[0])

    save_images(outputs, contents_path, output_dir, suffix=suffix)

    return outputs
