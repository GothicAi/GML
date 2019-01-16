from __future__ import print_function

from train_e_d import train_e_d
from train_facelet import train_facelet
from utils import list_images
#import tensorflow.contrib.eager as tfe

#tfe.enable_eager_execution()

is_training = True

# for training
TRAINING_CONTENT_DIR = 'MS_COCO'
ENCODER_WEIGHTS_PATH = 'vgg19_normalised.npz'
LOGGING_PERIOD = 20
feature_weight = 1
MODEL_SAVE_PATHS = [
    'models/style_weight_2e0.ckpt',
]

# for inferring (stylize)
INFERRING_CONTENT_DIR = 'images/content'
OUTPUTS_DIR = 'outputs'


def main():

    if is_training:

        content_imgs_path = list_images(TRAINING_CONTENT_DIR)

        train_facelet(content_imgs_path,"", ENCODER_WEIGHTS_PATH, MODEL_SAVE_PATHS)
        print('\n>>> Successfully! Done all training...\n')

    else:

        #content_imgs_path = list_images(INFERRING_CONTENT_DIR)

            #stylize(content_imgs_path, style_imgs_path, OUTPUTS_DIR, 
             #       ENCODER_WEIGHTS_PATH, model_save_path, 
             #       suffix='-' + str(style_weight))

        print('\n>>> Successfully! Done all stylizing...\n')


if __name__ == '__main__':
    main()