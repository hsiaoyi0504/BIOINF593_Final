import os

os.environ['KERAS_BACKEND'] = 'tensorflow'

import argparse
import time

import numpy as np
import pandas as pd
import tensorflow as tf
from keras import backend as K
from keras.backend.tensorflow_backend import set_session

from utils.image_pool import ImagePool
from models.discriminator import n_layer_discriminator
from models.generator import resnet_generator
from models.data_loader import (load_data,
                                minibatchAB)
from models.networks_utils import (get_generator_function,
                                   get_generator_outputs)
from models.train_function import (generator_train_function_creator,
                                   discriminator_A_train_function_creator,
                                   discriminator_B_train_function_creator,
                                   clip_weights)

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
set_session(sess)


def create_networks(network_type, generator_params, discriminator_params):
    netG_A, real_A, fake_B = resnet_generator(network_type=network_type, **generator_params)
    netG_B, real_B, fake_A = resnet_generator(network_type=network_type, **generator_params)

    netD_A = n_layer_discriminator(network_type=network_type, **discriminator_params)
    netD_B = n_layer_discriminator(network_type=network_type, **discriminator_params)

    discriminators_tuple = (netD_A, netD_B)
    generators_tuple = (netG_A, netG_B)
    real_imgs_tuple = (real_A, real_B)
    fake_imgs_tuple = (fake_A, fake_B)

    return discriminators_tuple, generators_tuple, real_imgs_tuple, fake_imgs_tuple


def create_generator_functions(generators_tuple):
    netG_A, netG_B = generators_tuple

    netG_A_function = get_generator_function(netG_A)
    netG_B_function = get_generator_function(netG_B)

    return netG_A_function, netG_B_function


def create_train_functions(discriminators_tuple,
                           generators_tuple,
                           real_imgs_tuple,
                           fake_imgs_tuple,
                           loss_weights_tuple,
                           input_shape,
                           use_wgan):
    netG_train_function = generator_train_function_creator(discriminators_tuple,
                                                           generators_tuple,
                                                           real_imgs_tuple,
                                                           fake_imgs_tuple,
                                                           loss_weights_tuple,
                                                           use_wgan)

    netD_A_train_function = discriminator_A_train_function_creator(discriminators_tuple,
                                                                   generators_tuple,
                                                                   real_imgs_tuple,
                                                                   input_shape,
                                                                   use_wgan)

    netD_B_train_function = discriminator_B_train_function_creator(discriminators_tuple,
                                                                   generators_tuple,
                                                                   real_imgs_tuple,
                                                                   input_shape,
                                                                   use_wgan)

    return netG_train_function, netD_A_train_function, netD_B_train_function


def create_image_pools(data_pool_size):
    fake_A_pool = ImagePool(pool_size=data_pool_size)
    fake_B_pool = ImagePool(pool_size=data_pool_size)

    return fake_A_pool, fake_B_pool


def create_batch_generators(data_path, train_file, test_file, input_shape, batch_size):
    train_A = load_data(os.path.join(data_path, train_file + 'A.csv'), input_shape)
    train_B = load_data(os.path.join(data_path, train_file + 'B.csv'), input_shape)
    train_batch = minibatchAB(train_A, train_B, batch_size=batch_size)

    test_A = load_data(os.path.join(data_path, test_file + 'A.csv'), input_shape)
    test_B = load_data(os.path.join(data_path, test_file + 'B.csv'), input_shape)
    test_batch = minibatchAB(test_A, test_B, batch_size=batch_size)

    batches_tuple = train_batch, test_batch
    test_data_tuple = test_A, test_B

    return batches_tuple, test_data_tuple


def save_networks(discriminators_tuple, generators_tuple, save_path):
    netD_A, netD_B = discriminators_tuple
    netG_A, netG_B = generators_tuple

    netG_A.save_weights(os.path.join(save_path, 'Generator_A_weights.h5'))
    netG_B.save_weights(os.path.join(save_path, 'Generator_B_weights.h5'))

    netD_A.save_weights(os.path.join(save_path, 'Discriminator_A_weights.h5'))
    netD_B.save_weights(os.path.join(save_path, 'Discriminator_B_weights.h5'))


def save_train_functions(train_functions_tuple, save_path):
    netG_train_function, netD_A_train_function, netD_B_train_function = train_functions_tuple

    netG_train_function.save_weights(os.path.join(save_path, 'Generator_train_function_weights.h5'))
    netD_A_train_function.save_weights(os.path.join(save_path, 'Discriminator_A_train_function_weights.h5'))
    netD_B_train_function.save_weights(os.path.join(save_path, 'Discriminator_B_train_function_weights.h5'))


def process_test_data(generators_tuple, test_data_tuple, save_path):
    netG_A, netG_B = generators_tuple
    netG_A.load_weights(os.path.join(save_path, 'Generator_A_weights.h5'))
    netG_B.load_weights(os.path.join(save_path, 'Generator_B_weights.h5'))
    test_A, test_B = test_data_tuple

    outputs = get_generator_outputs(netG_B, netG_A, test_B)
    fake_output, rec_input = outputs
    df_fake_output = pd.DataFrame(fake_output)
    df_fake_output.to_csv(os.path.join(save_path, 'FDA_cycle_GAN_encoded_n25_B_to_A.csv'))

    outputs = get_generator_outputs(netG_A, netG_B, test_A)
    fake_output, rec_input = outputs
    df_fake_output = pd.DataFrame(fake_output)
    df_fake_output.to_csv(os.path.join(save_path, 'FDA_cycle_GAN_encoded_n25_A_to_B.csv'))


def get_networks_params(input_shape, use_dropout, use_batch_norm, use_leaky_relu, use_wgan):
    generator_params = {
        'input_shape': input_shape,
        'use_dropout': use_dropout,
        'use_batch_norm': use_batch_norm,
        'use_leaky_relu': use_leaky_relu,
    }

    discriminator_params = {
        'input_shape': input_shape,
        'use_wgan': use_wgan,
        'use_batch_norm': use_batch_norm,
        'use_leaky_relu': use_leaky_relu,
    }

    return generator_params, discriminator_params


def train_model(network_parameters_tuple,
                loss_weights_tuple,
                train_settings_tuple,
                batches_tuple,
                test_data_tuple,
                generator_params,
                discriminator_params,
                saving_tuple):
    network_type, input_shape, use_wgan, data_pool_size = network_parameters_tuple
    save_path, save_model = saving_tuple

    K.set_learning_phase(1)

    discriminators_tuple, generators_tuple, real_imgs_tuple, fake_imgs_tuple = \
        create_networks(network_type, generator_params, discriminator_params)

    train_functions_tuple = \
        create_train_functions(discriminators_tuple,
                               generators_tuple,
                               real_imgs_tuple,
                               fake_imgs_tuple,
                               loss_weights_tuple,
                               input_shape,
                               use_wgan)

    generator_functions_tuple = create_generator_functions(generators_tuple)

    image_pools_tuple = create_image_pools(data_pool_size)
    '''
    run_train_loop(train_settings_tuple,
                   train_functions_tuple,
                   generator_functions_tuple,
                   image_pools_tuple,
                   batches_tuple,
                   discriminators_tuple)
    '''
    process_test_data(generators_tuple, test_data_tuple, save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--network_type", default="FC_smallest")
    parser.add_argument("--input_shape", default=56, type=int)

    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--epochs", default=100, type=int)
    parser.add_argument("--d_iters", default=5, type=int)
    parser.add_argument("--discriminator_patience", default=1, type=int)

    parser.add_argument("--use_wgan", default=False, type=bool)
    parser.add_argument("--use_batch_norm", default=True, type=bool)
    parser.add_argument("--use_leaky_relu", default=True, type=bool)
    parser.add_argument("--use_dropout", default=False, type=bool)
    parser.add_argument("--use_data_pooling", default=False, type=bool)

    parser.add_argument("--cycle_loss_weight", default=.3, type=float)
    parser.add_argument("--id_loss_weight", default=.1, type=float)
    parser.add_argument("--data_pool_size", default=500, type=int)

    parser.add_argument("--data_path", default="data/input_data/rule_of_5/")
    parser.add_argument("--train_file", default="rof_JTVAE_train_")
    parser.add_argument("--test_file", default="FDA_JTVAE_interpolation_i2_n25_")

    parser.add_argument("--save_path", default="data/results/rule_of_5/")
    parser.add_argument("--save_model", default=True, type=bool)

    parser.add_argument("--print_cost", default=True, type=bool)

    args = parser.parse_args()
    
    input_shape = (args.input_shape,)

    network_parameters_tuple = (args.network_type, input_shape, args.use_wgan, args.data_pool_size)
    saving_tuple = (args.save_path, args.save_model)

    loss_weights_tuple = (args.cycle_loss_weight, args.id_loss_weight)

    train_settings_tuple = (args.batch_size, args.epochs, args.d_iters,
                            args.discriminator_patience, args.use_data_pooling,
                            args.use_wgan, args.print_cost)

    batches_tuple, test_data_tuple = \
        create_batch_generators(args.data_path, args.train_file, args.test_file,
                                input_shape, args.batch_size)

    generator_params, discriminator_params = \
        get_networks_params(input_shape, args.use_dropout, args.use_batch_norm,
                            args.use_leaky_relu, args.use_wgan)

    train_model(network_parameters_tuple,
                loss_weights_tuple,
                train_settings_tuple,
                batches_tuple,
                test_data_tuple,
                generator_params,
                discriminator_params,
                saving_tuple)


if __name__ == "__main__":
    main()
