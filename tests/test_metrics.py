# -*- coding: utf-8 -*-
"""
Tests unitaires sur les fonctions d'évaluation.
"""
from statistics import mean
from unittest import TestCase, main

from src.addons.metrics import (
    average_precision,
    first_rank_accuracy,
    mean_average_precision,
    mean_reciprocal_rank,
    reciprocal_rank,
)


class TestMetric(TestCase):
    """
    Tests unitaires de la fonction d'évaluation.
    """

    def test_average_precision(self):
        avg_prec = average_precision(["one", "two", "one", "one", "two"], "one")
        real_avg_prec = mean([1, 2 / 3, 3 / 4])
        self.assertAlmostEqual(avg_prec, real_avg_prec)

    def test_mean_average_precision(self):
        maps = mean_average_precision(
            [["one", "two", "one", "one", "two"], ["one", "two", "one", "one", "two"]], ["one", "two"]
        )
        real_map = mean([mean([1, 2 / 3, 3 / 4]), mean([1 / 2, 2 / 5])])
        self.assertEqual(maps, real_map)

    def test_reciprocal_rank(self):
        self.assertEqual(1.0, reciprocal_rank(["one", "two", "one", "one", "two"], "one"))
        self.assertEqual(1 / 2, reciprocal_rank(["one", "two", "one", "one", "two"], "two"))
        self.assertEqual(1 / 3, reciprocal_rank(["two", "two", "one", "one", "two"], "one"))

    def test_mean_reciprocal_rank(self):
        mrr = mean_reciprocal_rank(
            [["one", "two", "one", "one", "two"], ["one", "two", "one", "one", "two"]], ["one", "two"]
        )
        real_mrr = mean([1, 1 / 2])
        self.assertEqual(mrr, real_mrr)

    def test_first_rank_accuracy(self):
        first_ranks = first_rank_accuracy(
            [["one", "two", "one", "one", "two"], ["one", "two", "one", "one", "two"]], ["one", "two"]
        )
        self.assertEqual(0.5, first_ranks)


if __name__ == "__main__":
    main()
