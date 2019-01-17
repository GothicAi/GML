from __future__ import print_function

from train_e_d import train_e_d
from utils import list_images
from infer import stylize
#import tensorflow.contrib.eager as tfe

#tfe.enable_eager_execution()
is_training = False

# for training
TRAINING_CONTENT_DIR = '/data/home/yingtai/tensorflow/p2p_makeup/data/plot_dist_data_0'
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
LOGGING_PERIOD = 20
feature_weight = 50
MODEL_SAVE_PATHS = [
    'Aug_L1_concat_models/encoder_decoder_concat_model.ckpt-2100',
]
#checkpoint = 'Aug_L1_concat_models'
# for inferring (stylize)
INFERRING_CONTENT_DIR = '/data/home/yingtai/tensorflow/p2p_makeup/data/validation_classdata'
OUTPUTS_DIR = 'outputs'


def main():

    if is_training:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)

        train_e_d(content_imgs_path, feature_weight, ENCODER_WEIGHTS_PATH, MODEL_SAVE_PATHS, checkpoint=None, logging_period=LOGGING_PERIOD, debug=False)

        print('\n>>> Successfully! Done all training...\n')

    else:

        content_imgs_path = list_images(INFERRING_CONTENT_DIR)

        stylize(content_imgs_path, OUTPUTS_DIR, ENCODER_WEIGHTS_PATH, MODEL_SAVE_PATHS, suffix=None)

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()
