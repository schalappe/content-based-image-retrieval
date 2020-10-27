# -*- coding: utf-8 -*-
import cv2
import pickle
import argparse
import progressbar
import numpy as np
from scripts import Extractor
from scripts import Retrievor
from preprocessors import AspectAwarePreprocessor
from preprocessors import ImageToArrayPreprocessor


# initialize process
aap = AspectAwarePreprocessor(224, 224)

image = cv2.imread('./data/test/105239817.jpg')
image = aap.preprocess(image)

extractor = Extractor("autoencoder")
retrievor = Retrievor('./features/autoencoder_features.pck')

features = extractor.extract(image)

distance = retrievor.search(features, depth=5)
print(distance)
