"""

Unit tests for multilabel_average_precision_metric.py

"""

import unittest
import numpy as np
from torch import Tensor

from relex.multilabel_average_precision_metric import MultilabelAveragePrecision
class TestUtil(unittest.TestCase):

    @classmethod
    def test_get_metrics(cls):
        np.seterr(divide='ignore', invalid='ignore')

        bins = 1000
        diff = 0.01
        metric = MultilabelAveragePrecision(bins=bins)
        size = [1000, 100]
        pred = Tensor(np.random.uniform(0, 1, size))
        gold = Tensor(np.random.randint(0, 2, size))
        metric.__call__(pred, gold)
        fast_ap = metric.get_metric()  # calls the fast get_metric
        ap = metric.get_metric(reset=True)  # calls the accurate get_metric
        assert (abs(ap - fast_ap)) < diff

        metric.__call__(pred, gold)
        metric.__call__(pred, gold)
        metric.__call__(pred, gold)
        fast_ap = metric.get_metric()
        ap = metric.get_metric(reset=True)
        assert (abs(ap - fast_ap)) < diff
