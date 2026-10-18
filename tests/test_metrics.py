# -*- coding: utf-8 -*-
"""
Tests unitaires sur les fonctions d'évaluation.
"""

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
    Tests unitaires des métriques tronquées à `k` résultats.
    """

    def test_average_precision_is_normalised_by_reachable_relevant_items(self):
        # ##: Five relevant items in the base, only the first result is one of them.
        self.assertAlmostEqual(
            average_precision(["one", "two", "two", "two", "two"], "one", relevant=100),
            1 / 5,
        )
        self.assertAlmostEqual(average_precision(["one", "two"], "one", relevant=2), 1 / 2)
        self.assertAlmostEqual(
            average_precision(["one", "two", "one", "one", "two"], "one", relevant=10),
            (1 + 2 / 3 + 3 / 4) / 5,
        )

    def test_average_precision_when_base_has_fewer_relevant_items_than_k(self):
        self.assertEqual(
            average_precision(["one", "two", "two", "two", "two"], "one", relevant=1),
            1.0,
        )

    def test_average_precision_without_hits_or_results(self):
        self.assertEqual(average_precision(["two", "two"], "one", relevant=3), 0.0)
        self.assertEqual(average_precision([], "one", relevant=3), 0.0)

    def test_mean_average_precision(self):
        retrievals = [["one", "one", "one"], ["one", "two", "one"]]
        self.assertAlmostEqual(
            mean_average_precision(retrievals, ["one", "two"], [50, 50]),
            (1 + (1 / 2) / 3) / 2,
        )

    def test_reciprocal_rank(self):
        self.assertEqual(reciprocal_rank(["one", "two", "one"], "one"), 1.0)
        self.assertEqual(reciprocal_rank(["one", "two", "one"], "two"), 1 / 2)
        self.assertEqual(reciprocal_rank(["one", "two"], "three"), 0.0)

    def test_failed_query_scores_zero(self):
        retrievals, labels = [["one"], []], ["one", "one"]
        self.assertEqual(mean_reciprocal_rank(retrievals, labels), 0.5)
        self.assertEqual(first_rank_accuracy(retrievals, labels), 0.5)
        self.assertEqual(mean_average_precision(retrievals, labels, [5, 5]), 0.5)

    def test_first_rank_accuracy(self):
        self.assertEqual(first_rank_accuracy([["one", "two"], ["one", "two"]], ["one", "two"]), 0.5)

    def test_mismatched_lengths_raise(self):
        with self.assertRaises(ValueError):
            mean_reciprocal_rank([["one"], ["two"]], ["one"])
        with self.assertRaises(ValueError):
            mean_average_precision([["one"]], ["one"], [1, 2])


if __name__ == "__main__":
    main()
