# -*- coding: utf-8 -*-
import os
import csv
import glob
import argparse
import progressbar
from sklearn.model_selection import train_test_split

ap = argparse.ArgumentParser()
ap.add_argument('-i', '--images', help='path of images')
args = vars(ap.parse_args())

# load images
imagePath = glob.glob(args['images']+'*/*.jpg')

train_set, test_set = train_test_split(
    imagePath, test_size=0.33, random_state=1331, shuffle=True
)

# parsing list
image_sets = [
    ('train', train_set, './outputs/train.csv'),
    ('test', test_set, './outputs/test.csv')
]

# loop over sets
for (dType, sets, outputPath) in image_sets:
    print('[INFO]: Building {} ...'.format(dType))
    # create file if necessary
    if not os.path.isfile(outputPath):
        with open(outputPath, 'w') as file:
            writer = csv.writer(file)
            writer.writerow(['path', 'color', 'type'])
    # progress bar
    widgets = [
        "Writing ...", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()
    ]
    pbar = progressbar.ProgressBar(
        maxval=len(sets), widgets=widgets
    ).start()
    for (i, path) in enumerate(sets):
        with open(outputPath, 'a+', newline='') as file:
            writer = csv.writer(file)
            color, style = path.split(os.path.sep)[-2].split('_')
            writer.writerow([path, color, style])
        pbar.update(i)
    pbar.finish()
