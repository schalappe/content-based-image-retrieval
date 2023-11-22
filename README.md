# Content based image retrieval

[Content-based image retrieval](https://en.wikipedia.org/wiki/Content-based_image_retrieval) (CBIR) is the application of computer vision techniques to the image retrieval problem, that is, the problem of searching for digital images in large databases.

**Content-base** means that the search analyzes the contents of the image rather than the metadata such as keywords, tags, or descriptions associated with the image.

![CBIR](reports/images/cbir.png)

It is carried out in three steps:
1. extraction of features from an image database to form a feature database,
2. extraction of the features of the input image,
3. find the most similar features in the database,
4. return the image associated with the found features

## Purpose

I would like to know which model and distance similarity is the most suitable for finding similar faces. For that, I try:

### Similarity measurement
* [Similarité cosinus](https://fr.wikipedia.org/wiki/Similarit%C3%A9_cosinus)
* [Distance de manhattan](https://fr.wikipedia.org/wiki/Distance_de_Manhattan)
* [Distance euclidienne](https://fr.wikipedia.org/wiki/Espace_euclidien)

### features extraction
* [AKAZE](https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html)
* [ORB](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)
* [VGG16](https://neurohive.io/en/popular-networks/vgg16/)
* [NasNet](https://paperswithcode.com/model/nasnet?variant=nasnetalarge)
* [EfficientNet](https://paperswithcode.com/method/efficientnet)

**The objective is to find the right combination (extraction algorithm & similarity measure) that allows to have relevant answers.**

## Part 1: Dataset

In my exploration, I used the following datasets:

* Fashion dataset [Apparel](https://www.kaggle.com/trolukovich/apparel-images-dataset)

## Part 2: Evaluation

CBIR system retrieves images based on feature similarity.
To evaluate my models, I used:

* Mean of Mean Average Precision (MMAP) for robustness of system
* Mean Reciprocal Rank (MRR) for the relevance of the first element
* average time per query

the evaluation formulas is referred to [here](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf)