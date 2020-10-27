# -*- coding: utf-8 -*-
import cv2
import pickle
import argparse
import progressbar
import numpy as np
from imutils import paths
from scripts import Extractor
from preprocessors import AspectAwarePreprocessor
from preprocessors import ImageToArrayPreprocessor


# arguments
ap = argparse.ArgumentParser()
ap.add_argument(
    '-t', '--type', required=True, help='type of extractor'
)
args = vars(ap.parse_args())

# extrator
iap = None
extractor = Extractor(args['type'])
if args['type'] in ['VGG16', 'autoencoder']:
    iap = ImageToArrayPreprocessor()

# data
imagePaths = list(paths.list_images('./data'))
# initialize process
aap = AspectAwarePreprocessor(224, 224)

# loop over images
db = []
widgets = [
    "Extract features: ", progressbar.Percentage(), " ",
    progressbar.Bar(), " ", progressbar.ETA()
]
pbar = progressbar.ProgressBar(maxval=len(imagePaths), widgets=widgets).start()
for i, image_path in enumerate(imagePaths):
    # preprocessing
    image = cv2.imread(image_path)
    image = aap.preprocess(image)
    if iap:
        image = iap.preprocess(image)
    features = extractor.extract(image)
    if isinstance(features, np.ndarray):
        db.append([features.flatten(), image_path])
    pbar.update(i)
pbar.finish()

with open('./features/'+args['type']+'_features.pck', 'wb') as fp:
    pickle.dump(db, fp)
print('Extraction finish. DB saved.')
