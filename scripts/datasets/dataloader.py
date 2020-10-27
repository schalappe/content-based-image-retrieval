# necessary packages
import os
import cv2
import glob
import numpy as np

from ..utils.utils import printProgressBar


class DatasetLoader:
    def __init__(self, preprocessors=None):
        # store the preprocessor
        self.preprocessors = preprocessors
        if self.preprocessors is None:
            self.preprocessors = []

    def load(self, path):
        # initialize the list
        imagePaths = sorted(glob.glob('%s/*.*' % path))
        length = len(imagePaths)
        data = []

        # loop
        printProgressBar(0, length, prefix='Progress:',
                         suffix='Complete', length=50)
        for (i, imagePath) in enumerate(imagePaths):
            image = cv2.imread(imagePath)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # preprocessing
            if self.preprocessors:
                for preprocessor in self.preprocessors:
                    image = preprocessor.preprocess(image)

            data.append(image)

            # show verbose
            printProgressBar(i + 1, length, prefix='Progress:',
                             suffix='Complete', length=50)

        return np.array(data)
