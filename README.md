# Content-based Image Retrieval

## Table of contents

1. [Description](#Description)
2. [Replication](#Replication)
3. [Result](#Result)
4. [License](#License)

## Description

[Content-based image retrieval](https://en.wikipedia.org/wiki/Content-based_image_retrieval) (CBIR) is the application of computer vision techniques to the image retrieval problem, which involves searching for digital images in large databases.

**Content-based** means that the search analyzes the contents of the image rather than metadata such as keywords, tags, or descriptions associated with the image.

![CBIR](reports/images/cbir.png)

The process consists of four steps:
1. Extraction of features from an image database to form a feature database.
2. Extraction of features from the input image.
3. Finding the most similar features in the database.
4. Returning the image associated with the found features.

### Purpose

The goal is to determine the most suitable model and distance similarity for finding similar images. To achieve this, we explore three different similarity measurements and five models for feature extraction.

#### Similarity Measurements
* [Cosine Similarity](https://en.wikipedia.org/wiki/Cosine_similarity)
* [Manhattan Distance](https://en.wikipedia.org/wiki/Manhattan_distance)
* [Euclidean Distance](https://en.wikipedia.org/wiki/Euclidean_space)

#### Feature Extraction
* [AKAZE](https://docs.opencv.org/3.4/db/d70/tutorial_akaze_matching.html)
* [ORB](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_orb/py_orb.html)
* [VGG16](https://neurohive.io/en/popular-networks/vgg16/)
* [NasNet](https://paperswithcode.com/model/nasnet?variant=nasnetalarge)
* [EfficientNet](https://paperswithcode.com/method/efficientnet)

**The objective is to find the right combination (extraction algorithm & similarity measure) that allows us to obtain relevant answers.**

### Evaluation

In our exploration, we used the Fashion dataset [Apparel](https://www.kaggle.com/trolukovich/apparel-images-dataset) available on Kaggle.
To evaluate different combinations (model + measurement), we utilized three metrics:

* Mean Average Precision (MAP) for the system's robustness.
* Mean Reciprocal Rank (MRR) for the relevance of the first element.
* Average time per query.

The evaluation formulas are referred to in [this](https://web.stanford.edu/class/cs276/handouts/EvaluationNew-handout-1-per.pdf) Stanford course.

## Replication

All the experiments can be reproduced using the Makefile:

- Create the necessary virtual environment for all tests:

```bash
make venv
```

- Create a repository for features and the .env file with different paths:

```bash
make prepare
```

- Create the feature dataset:

```bash
make features
```

## Result

Experimental results are presented in the report folder. Consult the PDF file and graphs to analyze the performance of different combinations.

## License

This project is licensed under the [MIT License](LICENSE).
