# necessary packages
import os
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import manhattan_distances
from sklearn.metrics.pairwise import euclidean_distances


class Retrievor:
    def __init__(self, compressor):
        if not os.path.isfile(compressor):
            raise ValueError("File of features doesn't exist")
        self.__load_compressor(compressor)

    def __load_compressor(self, compressor):
        with open(compressor, 'rb') as fp:
            features = pickle.load(fp)
        names = [f[1] for f in features]
        matrix = [f[0] for f in features]
        self.matrix = np.array(matrix)
        self.names = np.array(names)

    def compute_distance(self, vector, distance='cosinus'):
        v = vector.reshape(1, -1)
        if distance == 'cosinus':
            return cosine_similarity(self.matrix, v)
        elif distance == 'manhattan':
            return manhattan_distances(self.matrix, v)
        elif distance == 'euclidean':
            return euclidean_distances(self.matrix, v)

    def search(self, wanted, distance='cosinus', depth=1):
        distances = self.compute_distance(wanted, distance).flatten()
        nearest_ids = np.argsort(distances)[:depth].tolist()
        return self.names[nearest_ids].tolist(), distances[nearest_ids].tolist()
