# -*- coding: utf-8 -*-
"""
Tests unitaires sur la recherche des plus proches voisins.
"""

from unittest import TestCase, main

import numpy as np

from src.addons.finder import MEASURES, search


class TestSearch(TestCase):
    """
    Tests unitaires de l'ordre des résultats.
    """

    def test_closest_first_for_every_measure(self):
        gallery = np.array([[0.0, 1.0], [1.0, 0.0], [1.0, 0.2]])
        for measure in MEASURES:
            indices, distances, _ = search(gallery, np.array([[1.0, 0.0]]), measure, depth=3)
            self.assertEqual(indices[0].tolist(), [1, 2, 0], measure)
            self.assertTrue(np.all(np.diff(distances[0]) >= 0), measure)

    def test_cosine_ranks_by_direction_not_magnitude(self):
        gallery, query = np.array([[1.0, 1.0], [10.0, 0.0]]), np.array([[1.0, 0.0]])
        self.assertEqual(search(gallery, query, "cosine", depth=1)[0][0].tolist(), [1])
        self.assertEqual(search(gallery, query, "euclidean", depth=1)[0][0].tolist(), [0])

    def test_chunking_does_not_change_results(self):
        rng = np.random.default_rng(0)
        gallery, queries = rng.random((20, 4)), rng.random((7, 4))
        whole = search(gallery, queries, "manhattan", depth=5)
        chunked = search(gallery, queries, "manhattan", depth=5, chunk=3)
        np.testing.assert_array_equal(whole[0], chunked[0])
        np.testing.assert_allclose(whole[1], chunked[1])


if __name__ == "__main__":
    main()
