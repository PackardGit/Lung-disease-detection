
from flask import Flask, request, jsonify,send_from_directory
import uuid
import os
from tensorflow.keras.models import load_model
import numpy as np
import os
from flask import Flask, request, redirect, url_for
from werkzeug.utils import secure_filename
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
import numpy as np
import lungs_finder as lf
from PIL import Image, ImageFile
from io import BytesIO
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet import preprocess_input
from tensorflow.keras.applications.mobilenet import decode_predictions
import tensorflow as tf
## limit memory for gpu ##
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_virtual_device_configuration(
            gpus[0],
            [tf.config.experimental.VirtualDeviceConfiguration(memory_limit=1024 * 2)])
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)




UPLOAD_FOLDER = 'C:/Users/Adrian/server'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])
IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224
IMAGE_CHANNELS = 3


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

model_croping = tf.keras.models.load_model('H:/Projekty/Dane/COVID-19 Radiography Database/unet_lung_seg_serial.hdf5')
model = tf.keras.models.load_model('H:/Projekty/Dane/COVID-19 Radiography Database/resnet_variation.h5')

@app.route('/', methods=['GET'])
def send_index():
    return send_from_directory('./www', "index.html")

@app.route('/<path:path>', methods=['GET'])
def send_root(path):
    return send_from_directory('./www', path)

@app.route('/api/image', methods=['POST'])
def upload_image():
  # check if the post request has the file part
  if 'image' not in request.files:
      return jsonify({'error':'No posted image. Should be attribute named image.'})
  file = request.files['image']

  # if user does not select file, browser also
  # submit a empty part without filename
  if file.filename == '':
      return jsonify({'error':'Empty filename submitted.'})
  if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      #file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      x = []
      ImageFile.LOAD_TRUNCATED_IMAGES = False
      image1 = Image.open(BytesIO(file.read()))
      image1.load()
      imgage2 = Image.new("RGB", image1.size)
      imgage2.paste(image1)
      img = np.array(imgage2) 
      img = img[:, :, ::-1].copy() 

      
      
      #making prediction#
    #first cropp image 
      img_gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
      image = cv2.equalizeHist(img_gray)
             

      image_c = img_gray
      image_c = cv2.resize(image_c, (512,512), interpolation = cv2.INTER_AREA)
      image_c = np.array(image_c) / 255
      image_c = image_c.astype('float32')
      image_to_predict = image_c.reshape(1, 512, 512,1)
      predict_img = model_croping.predict(image_to_predict)
      mask = predict_img[0,:,:,0]

      mask = cv2.medianBlur(mask, 3)

      mask = skimage.morphology.opening(mask,disk(10))
      mask = skimage.morphology.dilation(mask, disk(20))
      mask = cv2.resize(mask, (224,224), interpolation = cv2.INTER_AREA)


      image_size = 224
      image = cv2.cvtColor(image,cv2.COLOR_GRAY2RGB)
      image = cv2.resize(image, (224,224), interpolation = cv2.INTER_AREA)
      x_test = np.array(image) / 255
      x_test = x_test.reshape(-1, image_size, image_size,3)
      x_test = x_test.astype('float32')

      dim = (image_size, image_size)
      # resize image
      image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)


      img_tensor = x_test
      layer = 'conv2d_5'
      conv_layer = model.get_layer(layer)

      heatmap_model = tf.keras.models.Model(
          [model.inputs], [model.get_layer(layer).output, model.output]
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

      prediction = model.predict(x_test)
      pred = np.argmax(prediction,axis=1)
      probability = prediction.max()*100
      result = 'unknown'
      if pred[0] == 0:
          result = 'Covid_19'
      if pred[0] == 1:
          result = 'Healthy Lungs!'
      if pred[0] == 2:
          result = 'Pneumonia'
        

      mask[mask > 0.5] = 1
      mask[mask <= 0.5] = 0
      result_image = image

      result_image[:,:,0] = result_image[:,:,0]*mask
      result_image[:,:,1] = result_image[:,:,1]*mask
      result_image[:,:,2] = result_image[:,:,2]*mask
      result_image[:,:,0]=image[:,:,1]*(1-heatmap) 
      result_image[:,:,1]=image[:,:,1]*heatmap 
      
      result_image2 = result_image
      result_image2[:,:,2] = result_image[:,:,0]
      result_image2[:,:,0] = result_image[:,:,1]
      ##########################    
      items = []
      items = {'name': result, 'prob': probability}
      response = {'pred':items}

      cv2.imwrite('C:/Users/Adrian/server/www/image.jpg', np.float32(result_image2)) 
      return jsonify(response) 
  else:
      return jsonify({'error':'File has invalid extension'})

if __name__ == '__main__':
    app.run(host= '0.0.0.0',debug=True)