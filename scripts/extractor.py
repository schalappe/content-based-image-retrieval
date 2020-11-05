# -*- coding: utf-8 -*-
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.models import load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications import MobileNetV2



class Extractor:
    def __init__(self, dsc_type, vector_size=32):
        self.size = vector_size
        self.type = dsc_type
        self.preprocess = None
        self.extractor = self.__create_descriptor()

    def __create_descriptor(self):
        if self.type == 'AKAZE':
            return cv2.AKAZE_create()
        if self.type == 'ORB':
            return cv2.ORB_create()
        if self.type == 'SURF':
            return cv2.xfeatures2d.SURF_create(400)
        if self.type == 'VGG16':
            self.preprocess = tf.keras.applications.vgg16.preprocess_input
            model = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            return Model(inputs=model.input, outputs=model.layers[-2].output)
        if self.type == 'VGG19':
            self.preprocess = tf.keras.applications.vgg19.preprocess_input
            model = VGG19(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            return Model(inputs=model.input, outputs=model.layers[-2].output)
        if self.type == 'MobileNet':
            self.preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
            model = MobileNetV2(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
            return Model(inputs=model.input, outputs=model.layers[-2].output)
        if self.type == 'autoencoder':
            return load_model('./outputs/encoder.h5')

    def descript(self, image):
        # Dinding image keypoints
        kps = self.extractor.detect(image)
        if not kps:
            return None
        # Getting first 32 of them.
        # Sorting them based on keypoint response value(bigger is better)
        kps = sorted(kps, key=lambda x: -x.response)[:self.size]
        # computing descriptors vector
        kps, dsc = self.extractor.compute(image, kps)
        # Flatten all of them in one big vector - our feature vector
        dsc = dsc.flatten()
        # Making descriptor of same size
        # Descriptor vector size is 64
        needed_size = (self.size * 64)
        if dsc.size < needed_size:
            # if we have less the 32 descriptors then just adding zeros at the
            # end of our feature vector
            dsc = np.concatenate([dsc, np.zeros(needed_size - dsc.size)])
        return np.array(dsc)

    def compress(self, image):
        if self.preprocess:
            image = self.preprocess(image)
        image = np.expand_dims(image, axis=0)
        return self.extractor.predict(image)[0].flatten()

    def extract(self, image):
        if self.type in ['ORB', 'AKAZE', 'SURF']:
            return self.descript(image)
        elif self.type in ['VGG16', 'VGG19', 'MobileNet', 'autoencoder',]:
            return self.compress(image)
