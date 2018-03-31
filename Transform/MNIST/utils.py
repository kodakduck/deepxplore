import random
from collections import defaultdict

import numpy as np
from keras import backend as K
from keras.models import Model

import cv2
from skimage.transform import rescale, resize
from skimage import transform as tf

img_rows, img_cols = 28, 28


# util function to convert a tensor into a valid image
def deprocess_image(x):
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    # original shape (1,img_rows, img_cols,1)
    return x.reshape(x.shape[1], x.shape[2])


def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)


def constraint_occl(gradients, start_point, rect_shape):
    # new_grads = np.ones_like(gradients)
    new_grads = np.zeros_like(gradients)
    new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
              start_point[1]:start_point[1] + rect_shape[1]] \
        = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                    start_point[1]:start_point[1] + rect_shape[1]]
    return new_grads


# def constraint_occl(gradients, start_point, rect_shape):
#     # new_grads = np.ones_like(gradients)
#     new_grads = np.zeros_like(gradients)
#     new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
#     start_point[1]:start_point[1] + rect_shape[1]] = np.ones_like(gradients)[:, start_point[0]:start_point[0] + rect_shape[0],
#                                                      start_point[1]:start_point[1] + rect_shape[1]]
#     return new_grads


def constraint_black(gradients, rect_shape=(6, 6)):
    start_point = (random.randint(0, gradients.shape[1] - rect_shape[0]),
                   random.randint(0, gradients.shape[2] - rect_shape[1]))
    new_grads = np.zeros_like(gradients)
    patch = gradients[:, start_point[0]:start_point[0] + rect_shape[0],
                      start_point[1]:start_point[1] + rect_shape[1]]
    # print(patch.shape)
    if np.mean(patch) < 0:
        new_grads[:, start_point[0]:start_point[0] + rect_shape[0],
                  start_point[1]:start_point[1] + rect_shape[1]] \
            = -np.ones_like(patch)
    return new_grads


def constraint_light(gradients):
    new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    print("grad_mean:", grad_mean)
    # print(new_grads.shape)
    return grad_mean * new_grads


def constraint_lightcontrast(gradients):
    grad_mean = np.mean(gradients)
    print("grad_mean:", 1 + grad_mean)
    return 1 + grad_mean


def constraint_darkcontrast(gradients):
    # new_grads = np.ones_like(gradients)
    grad_mean = np.mean(gradients)
    print("grad_mean:", grad_mean)
    # print(new_grads.shape)
    return grad_mean


def constraint_translate(image, gradients, step):
    img_shape = image.shape
    trans = np.mean(gradients) * step
    print("translate pixels:", trans)
    M = np.float32([[1, 0, trans], [0, 1, trans]])
    image = image.reshape((img_rows, img_cols, 1))
    gen_img = cv2.warpAffine(image, M, (img_rows, img_cols))
    return gen_img.reshape(img_shape)


def constraint_rotate(image, gradients, step):
    img_shape = image.shape
    print("rotate degrees:", step * np.mean(gradients))
    M = cv2.getRotationMatrix2D((img_rows / 2, img_cols / 2),
                                step * np.mean(gradients), 1)
    image = image.reshape((img_rows, img_cols, 1))
    gen_img = cv2.warpAffine(image, M, (img_rows, img_cols))
    return gen_img.reshape(img_shape)


def constraint_scale(image, gradients, step):
    img_shape = image.shape
    scale = min(abs(step * np.mean(gradients)), 1.2)
    print("scale:", scale)
    # if scale > 1.2:
    #     scale = 1.2
    image = image.reshape((img_rows, img_cols, 1))
    # dim = ((int)(img_rows * 1.1), (int)(img_cols * 1.1))
    # gen_img = cv2.cv2.resize(image, dim, interpolation=cv2.INTER_AREA)
    # gen_img = cv2.cv2.resize(image, (28, 28), interpolation=cv2.INTER_AREA)
    # gen_img = rescale(image, 1.1)
    # gen_img = resize(image, (28, 28))
    M = cv2.getRotationMatrix2D((img_rows / 2, img_cols / 2), 1, scale)
    gen_img = cv2.warpAffine(image, M, (img_rows, img_cols))
    return gen_img.reshape(img_shape)


def constraint_shear(image, gradients, step):
    img_shape = image.shape
    image = image.reshape((img_rows, img_cols, 1))
    shear = min(step * np.mean(gradients), 0.2)
    print("shar:", shear)
    # if scale > 1.2:
    #     scale = 1.2
    # Create Afine transform
    M = tf.AffineTransform(shear=shear)
    # Apply transform to image data
    gen_img = tf.warp(image, inverse_map=M)
    return gen_img.reshape(img_shape)


def init_coverage_tables(model1, model2, model3):
    model_layer_dict1 = defaultdict(bool)
    model_layer_dict2 = defaultdict(bool)
    model_layer_dict3 = defaultdict(bool)
    init_dict(model1, model_layer_dict1)
    init_dict(model2, model_layer_dict2)
    init_dict(model3, model_layer_dict3)
    return model_layer_dict1, model_layer_dict2, model_layer_dict3


def init_dict(model, model_layer_dict):
    for layer in model.layers:
        if 'flatten' in layer.name or 'input' in layer.name:
            continue
        # print("%s: %s" % (layer.name, layer.output_shape))
        for index in range(layer.output_shape[-1]):
            model_layer_dict[(layer.name, index)] = False


def neuron_to_cover(model_layer_dict):
    not_covered = [(layer_name, index) for (layer_name, index), v in model_layer_dict.items() if not v]
    # print(not_covered)
    if not_covered:
        layer_name, index = random.choice(not_covered)
    else:
        layer_name, index = random.choice(model_layer_dict.keys())
    print((layer_name, index))
    return layer_name, index


def neuron_covered(model_layer_dict):
    covered_neurons = len([v for v in model_layer_dict.values() if v])
    print("covered_neurons:", covered_neurons)
    total_neurons = len(model_layer_dict)
    print("total_neurons：", total_neurons)
    return covered_neurons, total_neurons, covered_neurons / float(total_neurons)


def update_coverage(input_data, model, model_layer_dict, threshold=0):
    layer_names = [layer.name for layer in model.layers if
                   'flatten' not in layer.name and 'input' not in layer.name]

    intermediate_layer_model = Model(inputs=model.input,
                                     outputs=[model.get_layer(layer_name).output for layer_name in layer_names])
    intermediate_layer_outputs = intermediate_layer_model.predict(input_data)

    for i, intermediate_layer_output in enumerate(intermediate_layer_outputs):
        scaled = scale(intermediate_layer_output[0])
        for num_neuron in range(scaled.shape[-1]):
            if np.mean(scaled[..., num_neuron]) > threshold and not model_layer_dict[(layer_names[i], num_neuron)]:
                model_layer_dict[(layer_names[i], num_neuron)] = True


def full_coverage(model_layer_dict):
    if False in model_layer_dict.values():
        return False
    return True


def scale(intermediate_layer_output, rmax=1, rmin=0):
    X_std = (intermediate_layer_output - intermediate_layer_output.min()) / (
        intermediate_layer_output.max() - intermediate_layer_output.min())
    X_scaled = X_std * (rmax - rmin) + rmin
    return X_scaled


def fired(model, layer_name, index, input_data, threshold=0):
    intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(layer_name).output)
    intermediate_layer_output = intermediate_layer_model.predict(input_data)[0]
    scaled = scale(intermediate_layer_output)
    if np.mean(scaled[..., index]) > threshold:
        return True
    return False


def diverged(predictions1, predictions2, predictions3, target):
    #     if predictions2 == predictions3 == target and predictions1 != target:
    if not predictions1 == predictions2 == predictions3:
        return True
    return False
