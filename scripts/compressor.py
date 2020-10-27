# -*- coding: utf-8 -*-
from tensorflow.keras.applications import VGG16


class Compressor:
    def __init__(self):
        self.extractor = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

    def extract(self, image):
        image = image.reshape(1, 224, 224, 3)
        return self.extractor.predict(image)[0]
