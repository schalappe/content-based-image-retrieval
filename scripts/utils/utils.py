# necessary packages
import os
import cv2
import numpy as np
from imutils import paths


def printProgressBar(iteration, total, prefix='', suffix='', decimals=1,
                     length=100, fill='█', printEnd="\r"):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in \
                        percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(
        100 * (iteration / float(total))
    )
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(
        '\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end=printEnd
    )
    # Print new Line on Complete
    if iteration == total:
        print()


def loadData(path, preprocessors=[]):
    # initialize the list
    imagePaths = list(paths.list_images(path))
    length = len(imagePaths)
    data, labels = [], []

    # loop
    printProgressBar(0, length, prefix='Progress:',
                     suffix='Complete', length=50)
    for (i, imagePath) in enumerate(imagePaths):
        image = cv2.imread(imagePath)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        label = imagePath.split(os.path.sep)[-2]
        # preprocessing
        if preprocessors:
            for preprocessor in preprocessors:
                image = preprocessor.preprocess(image)

        data.append(image)
        labels.append(label)

        # show verbose
        printProgressBar(i + 1, length, prefix='Progress:',
                         suffix='Complete', length=50)

    return (np.array(data), np.array(labels))
