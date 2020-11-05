# -*- coding: utf-8 -*-
import cv2
import pickle
import progressbar
import numpy as np
import pandas as pd
from scripts import Extractor
from preprocessors import AspectAwarePreprocessor
from preprocessors import ImageToArrayPreprocessor

# data
data = pd.read_csv('./outputs/train.csv')
# initialize process
iap = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(224, 224)

# loop over types
types = [
    #'AKAZE', 'ORB', 'SURF',
    #'VGG16', 'VGG19', 'MobileNet',
    'autoencoder'
]

# loop over images
for dType in types:
    print('[INFO]: Working with {} ...'.format(dType))
    extractor = Extractor(dType)
    db = []
    widgets = [
        "Extract features: ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()
    ]
    pbar = progressbar.ProgressBar(maxval=len(data), widgets=widgets).start()
    for index, row in data.iterrows():
        # preprocessing
        image = cv2.imread(row.path)
        image = aap.preprocess(image)
        if dType in ['VGG16', 'VGG19', 'MobileNet', 'autoencoder']:
            image = iap.preprocess(image)
        features = extractor.extract(image)
        if isinstance(features, np.ndarray):
            db.append([features, row.color, row.type])
        pbar.update(index)
    pbar.finish()

    with open('./features/' + dType + '_features.pck', 'wb') as fp:
        pickle.dump(db, fp)
    print('Extraction finish. DB saved.')
