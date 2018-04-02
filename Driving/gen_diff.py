'''
usage: python gen_diff.py -h
'''

# %run gen_diff.py light 1 0.1 10 20 50 0

from __future__ import print_function

import warnings
# h5py will issue a warning about deprecated np.float, ignore it
warnings.filterwarnings(action='ignore', category=FutureWarning)
import os.path
import argparse

from scipy.misc import imsave

from driving_models import *
from utils import *
import cv2
import numpy as np

import tensorflow as tf
# On windows, currently tensorflow does not allocate all available memory like it says in the documentation
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.Session(config=config)

import sys
import logging
log = logging.getLogger()
log.setLevel(logging.INFO)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
log.addHandler(ch)

# read the parameter
# argument parsing
parser = argparse.ArgumentParser(
    description='Main function for difference-inducing input generation in Driving dataset')
parser.add_argument('-transformation', help="realistic transformation type",
                    choices=['light', 'occl', 'blackout', 'contrast', 'rotate',
                             'translate', 'scale', 'shear', 'darkcontrast',
                             'sunlight', 'rain', 'water', 'fog', 'blur'])
parser.add_argument('-weight_diff', help="weight hyperparm to control differential behavior", type=float)
parser.add_argument('-weight_nc', help="weight hyperparm to control neuron coverage", type=float)
parser.add_argument('-step', help="step size of gradient descent", type=float)
parser.add_argument('-seeds', help="number of seeds of input", type=int)
parser.add_argument('-grad_iterations', help="number of iterations of gradient descent", type=int)
parser.add_argument('-threshold', help="threshold for determining neuron activated", type=float)
parser.add_argument('-t', '--target_model', help="target model that we want it predicts differently",
                    choices=[0, 1, 2], default=0, type=int)
parser.add_argument('-sp', '--start_point', help="occlusion upper left corner coordinate", default=(0, 0), type=tuple)
parser.add_argument('-occl_size', '--occlusion_size', help="occlusion size", default=(50, 50), type=tuple)
parser.add_argument('-eg1', help="example", default=True)

args = parser.parse_args()
if args.eg1:
    args.transformation = 'blur'
    args.weight_diff = 1
    args.weight_nc = 0.1
    args.step = 10
    args.seeds = 10
    args.grad_iterations = 100
    args.threshold = 0.8

# input image dimensions
img_rows, img_cols = 100, 100
input_shape = (img_rows, img_cols, 3)

# define input tensor as a placeholder
input_tensor = Input(shape=input_shape)

# load multiple models sharing same input tensor
K.set_learning_phase(0)
model1 = Dave_orig(input_tensor=input_tensor, load_weights=True)
model2 = Dave_norminit(input_tensor=input_tensor, load_weights=True)
model3 = Dave_dropout(input_tensor=input_tensor, load_weights=True)
# init coverage table
model_layer_dict1, model_layer_dict2, model_layer_dict3 = init_coverage_tables(model1, model2, model3)

# ==============================================================================================
# start gen inputs
img_paths = image.list_pictures('./testing/center', ext='jpg')
counter = 0
for _ in range(args.seeds):
    counter += 1
    orig_path = random.choice(img_paths)
    gen_img = preprocess_image(orig_path)
    orig_img = gen_img.copy()
    # first check if input already induces differences
    angle1, angle2, angle3 = model1.predict(gen_img)[0], model2.predict(gen_img)[0], model3.predict(gen_img)[0]
    if angle_diverged(angle1, angle2, angle3):
        print(bcolors.OKGREEN +
              'WITHOUT_TRAINING_input already causes different outputs: {}, {}, {}'.format(angle1, angle2,
                                                                                            angle3) + bcolors.ENDC)

        update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
        update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
        update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

        print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
              % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                 neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                 neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
        averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                       neuron_covered(model_layer_dict3)[0]) / float(
            neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
            neuron_covered(model_layer_dict3)[
                1])
        print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

        gen_img_deprocessed = draw_arrow(deprocess_image(gen_img), angle1, angle2, angle3)

        # save the result to disk
        imsave('./generated_inputs/' + 'WITHOUT_TRAINING_already_differ_' + str(angle1) + '_' + str(angle2) + '_' + str(angle3) + '.png',
               gen_img_deprocessed)
        continue

    # if all turning angles roughly the same
    orig_angle1, orig_angle2, orig_angle3 = angle1, angle2, angle3
    layer_name1, index1 = neuron_to_cover(model_layer_dict1)
    layer_name2, index2 = neuron_to_cover(model_layer_dict2)
    layer_name3, index3 = neuron_to_cover(model_layer_dict3)

    # construct joint loss function
    if args.target_model == 0:
        loss1 = -args.weight_diff * K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss2 = K.mean(model2.get_layer('before_prediction').output[..., 0])
        loss3 = K.mean(model3.get_layer('before_prediction').output[..., 0])
    elif args.target_model == 1:
        loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss2 = -args.weight_diff * K.mean(model2.get_layer('before_prediction').output[..., 0])
        loss3 = K.mean(model3.get_layer('before_prediction').output[..., 0])
    elif args.target_model == 2:
        loss1 = K.mean(model1.get_layer('before_prediction').output[..., 0])
        loss2 = K.mean(model2.get_layer('before_prediction').output[..., 0])
        loss3 = -args.weight_diff * K.mean(model3.get_layer('before_prediction').output[..., 0])
    loss1_neuron = K.mean(model1.get_layer(layer_name1).output[..., index1])
    loss2_neuron = K.mean(model2.get_layer(layer_name2).output[..., index2])
    loss3_neuron = K.mean(model3.get_layer(layer_name3).output[..., index3])
    layer_output = (loss1 + loss2 + loss3) + args.weight_nc * (loss1_neuron + loss2_neuron + loss3_neuron)

    # for adversarial image generation
    final_loss = K.mean(layer_output)

    # we compute the gradient of the input picture wrt this loss
    grads = normalize(K.gradients(final_loss, input_tensor)[0])

    # this function returns the loss and grads given the input picture
    iterate = K.function([input_tensor], [loss1, loss2, loss3, loss1_neuron, loss2_neuron, loss3_neuron, grads])

    # we run gradient ascent for 20 steps
    for iters in range(args.grad_iterations):
        loss_value1, loss_value2, loss_value3, loss_neuron1, loss_neuron2, loss_neuron3, grads_value = iterate(
            [gen_img])
        if args.transformation == 'light':
            grads_value = constraint_light(grads_value)  # constraint the gradients value
            gen_img += grads_value * args.step
        elif args.transformation == 'occl':
            grads_value = constraint_occl(grads_value, args.start_point,
                                          args.occlusion_size)  # constraint the gradients value
            gen_img += grads_value * args.step
        elif args.transformation == 'blackout':
            grads_value = constraint_black(grads_value)  # constraint the gradients value
            gen_img += grads_value * args.step
        elif args.transformation == 'sunlight':
            # We load the images
            camera_img = cv2.imread(orig_path.replace('/', '\\'), -1)
            # camera_img = cv2.imread(random.choice(img_paths), -1)
            camera_img = cv2.resize(camera_img, (100, 100))
            overlay_img = cv2.imread(".\\sunlight.png", -1)
            overlay_img = cv2.resize(overlay_img, (100, 100))
            rect = [52, 91, 24, 64]  # row: max, min; column: max, min
            gen_img = blend_transparent(camera_img, overlay_img,
                                        grads_value, rect)
            cv2.imwrite("merged.png", gen_img)
            gen_img = preprocess_image("merged.png")
        elif args.transformation == 'rain':
            # We load the images
            camera_img = cv2.imread(orig_path.replace('/', '\\'), -1)
            # camera_img = cv2.imread(random.choice(img_paths), -1)
            camera_img = cv2.resize(camera_img, (100, 100))
            overlay_img = cv2.imread(".\\rain.png", -1)
            overlay_img = cv2.resize(overlay_img, (100, 100))
            rect = [0, 100, 0, 100]  # row: max, min; column: max, min
            gen_img = blend_transparent(camera_img, overlay_img,
                                        grads_value, rect)
            cv2.imwrite("merged.png", gen_img)
            gen_img = preprocess_image("merged.png")
        elif args.transformation == 'water':
            # We load the images
            camera_img = cv2.imread(orig_path.replace('/', '\\'), -1)
            # camera_img = cv2.imread(random.choice(img_paths), -1)
            camera_img = cv2.resize(camera_img, (100, 100))
            overlay_img = cv2.imread(".\\water.png", -1)
            overlay_img = cv2.resize(overlay_img, (100, 100))
            rect = [0, 100, 0, 100]  # row: max, min; column: max, min
            gen_img = blend_transparent(camera_img, overlay_img,
                                        grads_value, rect)
            cv2.imwrite("merged.png", gen_img)
            gen_img = preprocess_image("merged.png")
        elif args.transformation == 'fog':
            # We load the images
            camera_img = cv2.imread(orig_path.replace('/', '\\'), -1)
            # camera_img = cv2.imread(random.choice(img_paths), -1)
            camera_img = cv2.resize(camera_img, (100, 100))
            overlay_img = cv2.imread(".\\fog.png", -1)
            overlay_img = cv2.resize(overlay_img, (100, 100))
            rect = [0, 100, 0, 100]  # row: max, min; column: max, min
            gen_img = blend_transparent(camera_img, overlay_img,
                                        grads_value, rect)
            cv2.imwrite("merged.png", gen_img)
            gen_img = preprocess_image("merged.png")
        elif args.transformation == 'blur':
            # We load the images
            camera_img = cv2.imread(orig_path.replace('/', '\\'), -1)
            # camera_img = cv2.imread(random.choice(img_paths), -1)
            camera_img = cv2.resize(camera_img, (100, 100))
            gen_img = constraint_blur(camera_img)
            cv2.imwrite("merged.png", gen_img)
            gen_img = preprocess_image("merged.png")

        angle1, angle2, angle3 = model1.predict(gen_img)[0], model2.predict(gen_img)[0], model3.predict(gen_img)[0]

        if angle_diverged(angle1, angle2, angle3):
            update_coverage(gen_img, model1, model_layer_dict1, args.threshold)
            update_coverage(gen_img, model2, model_layer_dict2, args.threshold)
            update_coverage(gen_img, model3, model_layer_dict3, args.threshold)

            print(bcolors.OKGREEN + '\nSUCCESS_Training complete after '+str(counter)+' iterations\n')
            print(bcolors.OKGREEN + 'covered neurons percentage %d neurons %.3f, %d neurons %.3f, %d neurons %.3f'
                  % (len(model_layer_dict1), neuron_covered(model_layer_dict1)[2], len(model_layer_dict2),
                     neuron_covered(model_layer_dict2)[2], len(model_layer_dict3),
                     neuron_covered(model_layer_dict3)[2]) + bcolors.ENDC)
            averaged_nc = (neuron_covered(model_layer_dict1)[0] + neuron_covered(model_layer_dict2)[0] +
                           neuron_covered(model_layer_dict3)[0]) / float(
                neuron_covered(model_layer_dict1)[1] + neuron_covered(model_layer_dict2)[1] +
                neuron_covered(model_layer_dict3)[
                    1])
            print(bcolors.OKGREEN + 'averaged covered neurons %.3f' % averaged_nc + bcolors.ENDC)

            gen_img_deprocessed = draw_arrow(deprocess_image(gen_img), angle1, angle2, angle3)
            orig_img_deprocessed = draw_arrow(deprocess_image(orig_img), orig_angle1, orig_angle2, orig_angle3)

            # save the result to disk
            imsave('./generated_inputs/SUCCESS_' + args.transformation + '_' + str(angle1) + '_' + str(angle2) + '_' + str(
                angle3) + '.png', gen_img_deprocessed)
            imsave('./generated_inputs/SUCCESS_' + args.transformation + '_' + str(angle1) + '_' + str(angle2) + '_' + str(
                angle3) + '_orig.png', orig_img_deprocessed)
            break
