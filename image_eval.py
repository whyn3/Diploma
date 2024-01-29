######################################################
#input shape must be NHWC shape as Tensorflow's Tensor
######################################################
import tensorflow as tf
import numpy as np
import cv2,os,math,keras,torch
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model
from scipy.fftpack import dct
from scipy.stats import skew, kurtosis
import cv2
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
import lpips,torch
from DISTS_pytorch import DISTS
loss_fn_alex = lpips.LPIPS(net='alex') # best forward scores
Ds = DISTS()
activate=1
if activate:
    from tensorflow.keras.applications import VGG16
    from tensorflow.keras.models import Model
    # Construct VGG16 model outside the loss function
    vgg = VGG16(include_top=False, input_shape=(None, None, 3))
    vgg.trainable = False
    vgg_output_layers = [vgg.get_layer('block1_conv2').output,
                        vgg.get_layer('block2_conv2').output,
                        vgg.get_layer('block3_conv3').output,
                        vgg.get_layer('block4_conv3').output]
    vgg_model = Model(inputs=vgg.input, outputs=vgg_output_layers)

def Perceptual_loss(y_true, y_pred):#perceptual loss only, for single input!!!!
    y_true = tf.cast(y_true, dtype='float32')
    y_pred = tf.cast(y_pred, dtype='float32')
    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))
    # Perceptual loss calculation using pre-constructed VGG16 model
    y_true_expanded = tf.image.grayscale_to_rgb(y_true)
    y_pred_expanded = tf.image.grayscale_to_rgb(y_pred)
    y_true_features = vgg_model(y_true_expanded)
    y_pred_features = vgg_model(y_pred_expanded)
    perceptual_loss = 0.0
    for true_feature, pred_feature in zip(y_true_features, y_pred_features):
        perceptual_loss += tf.reduce_mean(tf.square(true_feature - pred_feature))
    perceptual_loss /= len(y_true_features)
    # Combine NPCC and perceptual loss
    return perceptual_loss.numpy()

def blinds_ii(image): #for grayscale
    image = (image - np.min(image)) / (np.max(image) - np.min(image))
    image = np.round(image * 255).astype(np.uint8)
    # Perform 8x8 block-wise DCT
    dct_blocks = dct(dct(image, axis=0, norm='ortho'), axis=1, norm='ortho')
    # Extract statistical features
    mean_dct = np.mean(dct_blocks)
    std_dct = np.std(dct_blocks)
    skew_dct = skew(dct_blocks.flatten())
    kurt_dct = kurtosis(dct_blocks.flatten())
    # Combine features into a quality estimate
    quality_estimate = mean_dct - 0.02 * std_dct + 0.03 * skew_dct - 0.01 * kurt_dct
    return quality_estimate

def calculate_vif(y_true, y_pred):
    y_true=tf.squeeze(y_true,axis=0)#make it HW1, so assuing single image
    y_pred=tf.squeeze(y_pred,axis=0)
    y_true = (y_true - np.min(y_true)) / (np.max(y_true) - np.min(y_true))
    y_pred = (y_pred - np.min(y_pred)) / (np.max(y_pred) - np.min(y_pred))
    y_true = (y_true * 255).numpy().astype(np.uint8)
    y_pred = (y_pred * 255).numpy().astype(np.uint8)  
    vif = cv2.matchTemplate(y_true, y_pred, cv2.TM_CCOEFF_NORMED)[0][0]
    return vif

def psnr(y_true, y_pred):
    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))
    if y_pred.shape[-1] != 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)
    x=tf.image.psnr(y_true,y_pred,max_val=1.0)
    return x.numpy()

def ssim(y_true, y_pred):
    y_true=tf.cast(y_true,dtype='float32')
    y_pred=tf.cast(y_pred,dtype='float32')
    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))
    if y_pred.shape[-1] != 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)
    x=tf.image.ssim(y_true,y_pred,max_val=1.0)
    return x.numpy()

def msssim(y_true, y_pred):
    y_true=tf.cast(y_true,dtype='float32')
    y_pred=tf.cast(y_pred,dtype='float32')
    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))
    if y_pred.shape[-1] != 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)
    x=tf.image.ssim_multiscale(y_true,y_pred,max_val=1.0)
    return x.numpy()

def psnr_hvs(img1, img2):
    img1 = (img1 - K.min(img1)) / (K.max(img1) - K.min(img1))
    img2 = (img2 - K.min(img2)) / (K.max(img2) - K.min(img2))
    mse = tf.reduce_mean(tf.square(img1 - img2))
    # Constants for PSNR-HVS calculation
    max_pixel_value = 255.0
    K1 = 0.01
    K2 = 0.03
    alpha = 1.0
    beta = 0.85

    # Calculate PSNR-HVS
    psnr_hvs = 20 * tf.math.log(max_pixel_value / tf.math.sqrt(mse)) - \
                alpha * (1 - tf.math.exp(-beta * (mse / (K1 * max_pixel_value))**2))
    return psnr_hvs

def NPCC(y_true, y_pred):#NPCC only
    y_true = (y_true - K.min(y_true)) / (K.max(y_true) - K.min(y_true))
    y_pred = (y_pred - K.min(y_pred)) / (K.max(y_pred) - K.min(y_pred))
    y_true=tf.cast(y_true,dtype='float32')
    y_pred=tf.cast(y_pred,dtype='float32')
    # Reshape the 2D images into 1D tensors
    image1_1d = tf.reshape(y_true, [-1])
    image2_1d = tf.reshape(y_pred, [-1])
    # Calculate the means of the images
    mean_image1 = tf.reduce_mean(image1_1d)
    mean_image2 = tf.reduce_mean(image2_1d)
    # Calculate the covariance between the two images
    covariance = tf.reduce_mean((image1_1d - mean_image1) * (image2_1d - mean_image2))
    # Calculate the standard deviations of the images
    stddev_image1 = tf.sqrt(tf.reduce_mean(tf.square(image1_1d - mean_image1)))
    stddev_image2 = tf.sqrt(tf.reduce_mean(tf.square(image2_1d - mean_image2)))
    # Calculate the Pearson correlation coefficient between the two images
    pearson_coefficient_loss = 1 - (covariance / (stddev_image1 * stddev_image2)) #NPCC=-1 is the best, it will converge to 0
    return pearson_coefficient_loss

def tensor_reshaper(input_tensor):
    # check input shape of tensorflow tensor and make it NHW1
    input_shape = input_tensor.shape
    # Check and convert to NHW1 format
    if len(input_shape) == 4 and input_shape[3] == 1:
        # Already in NHW1 format, no need to convert
        return input_tensor
    elif len(input_shape) == 3 and input_shape[2] == 1:
        # Convert from HW1 to NHW1
        return tf.expand_dims(input_tensor, axis=0)
    elif len(input_shape) == 3 and input_shape[2] != 1:
        # Convert from NHW to NHW1
        return tf.expand_dims(input_tensor, axis=-1)
    elif len(input_shape) == 2:
        # Convert from HW to NHW1
        return tf.expand_dims(tf.expand_dims(input_tensor, axis=0), axis=-1)
    else:
        raise ValueError("Unsupported input shape")