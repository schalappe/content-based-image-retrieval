# -*- coding: utf-8 -*-
import os
import cv2
import csv
import numpy as np
import pandas as pd
import progressbar
from scripts import Timer
from scripts import Retrievor
from scripts import Extractor
from scripts import mean_reciprocal_rank
from scripts import mean_mean_average_precision
from scripts import rank1_accuracy
from preprocessors import AspectAwarePreprocessor
from preprocessors import ImageToArrayPreprocessor

# data
data = pd.read_csv('./outputs/test.csv')
# initialize process
iap = ImageToArrayPreprocessor()
aap = AspectAwarePreprocessor(224, 224)

# evaluation.csv
if not os.path.isfile('./outputs/evaluation.csv'):
    with open('./outputs/evaluation.csv', 'w') as file:
        writer = csv.writer(file)
        writer.writerow([
            'extractor', 'distance', 'mrr', 'mmap', 'accuracy', 'average_time', 'errors', 'comparison_item'
        ])

# type of extractor
types = [
    'ORB', 'SURF', 'AKAZE',
    'VGG16', 'VGG19', 'MobileNet',
    'autoencoder'
]
# distances
distances = [
    'cosinus', 'manhattan', 'euclidean'
]
# metric
items = [
    'color', 'style', 'both'
]

for dType in types:
    db, errors = [], 0
    colors, styles = [], []
    both = []
    extractor = Extractor(dType)
    timer = Timer()
    print('[INFO]: Working on {} ...'.format(dType))
    widgets = [
        "Evaluation: PART 1 - Extraction ", progressbar.Percentage(), " ",
        progressbar.Bar(), " ", progressbar.ETA()
    ]
    pbar = progressbar.ProgressBar(maxval=len(data), widgets=widgets).start()
    # loop over data
    for index, row in data.iterrows():
        # preprocessing
        timer.tic()
        image = cv2.imread(row.path)
        image = aap.preprocess(image)
        if dType in ['VGG16', 'VGG19', 'MobileNet', 'autoencoder']:
            image = iap.preprocess(image)
        features = extractor.extract(image)
        timer.toc()
        if not isinstance(features, np.ndarray):
            errors += 1
            continue
        db.append(features)
        colors.append(row.color)
        styles.append(row.type)
        both.append(row.color + '_' + row.type)
        pbar.update(index)
    pbar.finish()
    average_time_extraction = timer.average_time

    for distance in distances:
        timer.clear()
        # varaibles
        retrievor = Retrievor('./features/' + dType + '_features.pck')
        retrievals_color, retrievals_style = [], []
        retrievals_both = []
        widgets = [
            "Evaluation: PART 2 - Search ", progressbar.Percentage(), " ",
            progressbar.Bar(), " ", progressbar.ETA()
        ]
        pbar = progressbar.ProgressBar(maxval=len(data), widgets=widgets).start()
        for i, features in enumerate(db):
            timer.tic()
            _colors, _styles, _ = retrievor.search(features, distance, depth=5)
            timer.toc()
            retrievals_color.append(_colors)
            retrievals_style.append(_styles)
            retrievals_both.append([
                c + '_' + s for c, s in zip(_colors, _styles)
            ])
            pbar.update(i)
        pbar.finish()
        # summary
        average_time_search = timer.average_time
        for item in items:
            if item == 'color':
                mrr = mean_reciprocal_rank(retrievals_color, colors)
                mmap = mean_mean_average_precision(retrievals_color, colors)
                rank_1 = rank1_accuracy(retrievals_color, colors)
            elif items == 'style':
                mrr = mean_reciprocal_rank(retrievals_style, styles)
                mmap = mean_mean_average_precision(retrievals_style, styles)
                rank_1 = rank1_accuracy(retrievals_style, styles)
            else:
                mrr = mean_reciprocal_rank(retrievals_both, both)
                mmap = mean_mean_average_precision(retrievals_both, both)
                rank_1 = rank1_accuracy(retrievals_both, both)
            with open('./outputs/evaluation.csv', 'a+', newline='') as file:
                writer = csv.writer(file)
                writer.writerow([
                    dType, distance, mrr, mmap, rank_1, average_time_search + average_time_extraction, errors, item
                ])
