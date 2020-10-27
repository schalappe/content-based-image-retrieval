# Content based image retrieval

[Content-based image retrieval][1] (CBIR) is the application of computer vision techniques to the image retrieval problem, that is, the problem of searching for digital images in large databases.

**Content-base** means that the search analyzes the contents of the image rather than the metadata such as keywords, tags, or descriptions associated with the image.

## Purpose

I would like to know which model and distance similarity is the most suitable for finding similar faces. For that, I try:

* [AKAZE][4]
* [ORB][5]
* [Autoencoder][6]

* Cosine similarity
* Manhattan distance
* Euclidean distance

## Part 1: Dataset

In my exploration, I used the following datasets:

* The real faces of [Real and Fake Face Detection][2] for training the autoencoder
* [5 Celebrity Faces Dataset][3] for evaluation

## Part 2: Evaluation

CBIR system retrieves images based on feature similarity.
To evaluate my models, I used:

* Mean of Mean Average Precision (MMAP)
* Mean Reciprocal Rank (MRR)
* Rank #1 Accuracy
* average time per query

the evaluation formulas is refer to [here][7]

[1]: https://en.wikipedia.org/wiki/Content-based_image_retrieval
[2]:https://www.kaggle.com/ciplab/real-and-fake-face-detection
[3]: https://www.kaggle.com/dansbecker/5-celebrity-faces-dataset
[4]: https://docs.opencv.org/4.2.0/db/d70/tutorial_akaze_matching.html
[5]: https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html
[6]: https://en.wikipedia.org/wiki/Autoencoder
[7]: https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf