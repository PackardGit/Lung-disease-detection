import base64
import cv2
import os, glob
import numpy as np
import seaborn as sns
from imutils import paths
import matplotlib.pyplot as plt
import sklearn.metrics as metrics
import skimage.morphology 
from skimage.morphology import disk
from tensorflow.keras import backend as K
import tensorflow as tf
import numpy as np
import lungs_finder as lf
from os.path import dirname, join
from tensorflow.keras.models import *
from tensorflow.keras.layers import *


def main(data):
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data,np.uint8)
    
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    image2 = cv2.equalizeHist(img_gray)
    image = cv2.medianBlur(image2, 3)
    scaled_image = image# cv2.resize(image, (1024,1024))
    right_lung_hog_rectangle = lf.find_right_lung_hog(scaled_image)
    left_lung_hog_rectangle = lf.find_left_lung_hog(scaled_image)
    right_lung_lbp_rectangle = lf.find_right_lung_lbp(scaled_image)
    left_lung_lbp_rectangle = lf.find_left_lung_lbp(scaled_image)
    right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
    left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)
    color_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

    if (right_lung_hog_rectangle is not None) and (left_lung_hog_rectangle is not None) :
        x1, y1, width1, height1 = right_lung_hog_rectangle
        x2, y2, width2, height2 = left_lung_hog_rectangle
        cimage1 = image2[np.minimum(y1, y2):np.maximum(y1+height1, y2+height2),x1:x2+width2] 
        pole1 = width1*height1+width2*height2

    if (right_lung_lbp_rectangle is not None) and (left_lung_lbp_rectangle is not None):
        x1, y1, width1, height1 = right_lung_lbp_rectangle
        x2, y2, width2, height2 = left_lung_lbp_rectangle              
        cimage2 = image2[np.minimum(y1, y2):np.maximum(y1+height1, y2+height2),x1:x2+width2] 
        pole2 = width1*height1+width2*height2

    if (right_lung_haar_rectangle is not None) and (left_lung_haar_rectangle is not None):
        x1, y1, width1, height1 = right_lung_haar_rectangle
        x2, y2, width2, height2 = left_lung_haar_rectangle
        cimage3 = image2[np.minimum(y1, y2):np.maximum(y1+height1, y2+height2),x1:x2+width2] 
        pole3 = width1*height1+width2*height2

    if (pole1 >= pole2) and (pole1 >= pole3) and (pole1>0):
        cimage = cimage1
    if (pole2 > pole1) and (pole2 >= pole3):
        cimage = cimage2
    if (pole3 > pole1) and (pole3 > pole2):
        cimage = cimage3
    if (cimage.shape[0]>0) and (cimage.shape[1]>0):
        image = cimage
        
    filename = join(dirname(__file__), "resnet_cropped_CvHvP_detection2.h5")
    model = tf.keras.models.load_model(filename)
    filename = join(dirname(__file__), "unet_lung_seg_serial.hdf5")
    model_croping = tf.keras.models.load_model(filename)
    
    ##
    
    image_c = cimage
    image_c = cv2.resize(image_c, (512,512), interpolation = cv2.INTER_AREA)
    image_c = np.array(image_c) / 255
    image_c = image_c.astype('float32')
    image_to_predict = image_c.reshape(1, 512, 512,1)
    predict_img = model_croping.predict(image_to_predict)
    mask = predict_img[0,:,:,0]

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0.35

    mask = cv2.medianBlur(mask, 3)
    #mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)

    #img = skimage.morphology.remove_small_objects(img,min_size=10000, connectivity=8)
    mask = skimage.morphology.opening(mask,disk(10))
    mask = skimage.morphology.dilation(mask, disk(20))
    mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)

    image_size = 224
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
    #image = np.random.random((image_size, image_size, 3))
    x_test = np.array(image) / 255
    x_test = x_test.reshape(-1, image_size, image_size,3)
    x_test = x_test.astype('float32')

    dim = (image_size, image_size)
    # resize image
    image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


    img_tensor = x_test

    conv_layer = model.get_layer('conv5_block3_2_conv')

    heatmap_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer('conv5_block3_2_conv').output, model.output]
    )

    with tf.GradientTape() as tape:
        conv_output, predictions = heatmap_model(img_tensor)
        loss = predictions[:, np.argmax(predictions[0])]

    grads = tape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

    heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
    heatmap = np.maximum(heatmap, 0)
    max_heat = np.max(heatmap)
    if max_heat == 0:
        max_heat = 1e-10
    heatmap /= max_heat
    heatmap = heatmap[0,:,:]
    heatmap = cv2.resize(heatmap, dim, interpolation = cv2.INTER_AREA)
    heatmap = 1 - heatmap

    result_image = image
    result_image[:,:,1]=image[:,:,1]*heatmap 
    result_image[:,:,2]=image[:,:,1]*heatmap 
    result_image[:,:,0] = result_image[:,:,0]*mask
    result_image[:,:,1] = result_image[:,:,1]*mask
    result_image[:,:,2] = result_image[:,:,2]*mask    

    pil_im = Image.fromarray(result_image)
    if pil_im.mode != 'RGB':
        pil_im = pil_im.convert('RGB')
    buff = io.BytesIO()
    pil_im.save(buff,format="PNG")
    img_str = base64.b64encode(buff.getvalue())
    return ""+str(img_str,'utf-8')

def pred(data):
    decoded_data = base64.b64decode(data)
    np_data = np.fromstring(decoded_data,np.uint8)
    
    img = cv2.imdecode(np_data,cv2.IMREAD_UNCHANGED)
    img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    image2 = cv2.equalizeHist(img_gray)
    image = cv2.medianBlur(image2, 3)
    scaled_image = image# cv2.resize(image, (1024,1024))
    right_lung_hog_rectangle = lf.find_right_lung_hog(scaled_image)
    left_lung_hog_rectangle = lf.find_left_lung_hog(scaled_image)
    right_lung_lbp_rectangle = lf.find_right_lung_lbp(scaled_image)
    left_lung_lbp_rectangle = lf.find_left_lung_lbp(scaled_image)
    right_lung_haar_rectangle = lf.find_right_lung_haar(scaled_image)
    left_lung_haar_rectangle = lf.find_left_lung_haar(scaled_image)
    color_image = cv2.cvtColor(scaled_image, cv2.COLOR_GRAY2BGR)

    if (right_lung_hog_rectangle is not None) and (left_lung_hog_rectangle is not None) :
        x1, y1, width1, height1 = right_lung_hog_rectangle
        x2, y2, width2, height2 = left_lung_hog_rectangle
        cimage1 = image2[np.minimum(y1, y2):np.maximum(y1+height1, y2+height2),x1:x2+width2] 
        pole1 = width1*height1+width2*height2

    if (right_lung_lbp_rectangle is not None) and (left_lung_lbp_rectangle is not None):
        x1, y1, width1, height1 = right_lung_lbp_rectangle
        x2, y2, width2, height2 = left_lung_lbp_rectangle              
        cimage2 = image2[np.minimum(y1, y2):np.maximum(y1+height1, y2+height2),x1:x2+width2] 
        pole2 = width1*height1+width2*height2

    if (right_lung_haar_rectangle is not None) and (left_lung_haar_rectangle is not None):
        x1, y1, width1, height1 = right_lung_haar_rectangle
        x2, y2, width2, height2 = left_lung_haar_rectangle
        cimage3 = image2[np.minimum(y1, y2):np.maximum(y1+height1, y2+height2),x1:x2+width2] 
        pole3 = width1*height1+width2*height2

    if (pole1 >= pole2) and (pole1 >= pole3) and (pole1>0):
        cimage = cimage1
    if (pole2 > pole1) and (pole2 >= pole3):
        cimage = cimage2
    if (pole3 > pole1) and (pole3 > pole2):
        cimage = cimage3
    if (cimage.shape[0]>0) and (cimage.shape[1]>0):
        image = cimage
        
    filename = join(dirname(__file__), "unet2_lung_seg_hist.hdf5")
    model_croping = keras.models.load_model(filename, custom_objects={'dice_coef_loss': dice_coef_loss,'dice_coef': dice_coef} )
    filename = join(dirname(__file__), "resnet_cropped_CvHvP_detection.hdf5'")
    model = keras.models.load_model(filename )
    ##
    
    image_c = cimage
    image_c = cv2.resize(image_c, (512,512), interpolation = cv2.INTER_AREA)
    image_c = np.array(image_c) / 255
    image_c = image_c.astype('float32')
    image_to_predict = image_c.reshape(1, 512, 512,1)
    predict_img = model_croping.predict(image_to_predict)
    mask = predict_img[0,:,:,0]

    mask[mask > 0.5] = 1
    mask[mask <= 0.5] = 0.35

    mask = cv2.medianBlur(mask, 3)
    #mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)

    #img = skimage.morphology.remove_small_objects(img,min_size=10000, connectivity=8)
    mask = skimage.morphology.opening(mask,disk(10))
    mask = skimage.morphology.dilation(mask, disk(20))
    mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)

    image_size = 224
    image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
    image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
    #image = np.random.random((image_size, image_size, 3))
    x_test = np.array(image) / 255
    x_test = x_test.reshape(-1, image_size, image_size,3)
    x_test = x_test.astype('float32')

    prediction = model.predict(x_test)
    pred = np.argmax(prediction,axis=1)
    probability = prediction.max()
    result = 'unknown'
    if pred[0] == 0:
        result = 'Covid_19'
    if pred[0] == 1:
        result = 'Healthy Lungs!'
    if pred[0] == 2:
        result = 'Pneumonia'
    probability = round(probability*100, 2)
    a = (f'The pattient has {result} with probability of {probability} %')   
    return a