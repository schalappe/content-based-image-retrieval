# -*- coding: utf-8 -*-
import cv2
import pickle
import argparse
import progressbar
import numpy as np
from src.scripts import Extractor
from src.scripts import Retrievor
from src.preprocessors import AspectAwarePreprocessor
from src.preprocessors import ImageToArrayPreprocessor


# initialize process
aap = AspectAwarePreprocessor(224, 224)

image = cv2.imread('./data/test/105239817.jpg')
image = aap.preprocess(image)

extractor = Extractor("autoencoder")
retrievor = Retrievor('./features/autoencoder_features.pck')

features = extractor.extract(image)

distance = retrievor.search(features, depth=5)
print(distance)
