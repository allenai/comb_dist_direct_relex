import logging
from overrides import overrides
import numpy as np
import torch
from torch import nn
from sklearn.metrics import precision_recall_curve
from allennlp.common.checks import ConfigurationError
from allennlp.training.metrics.metric import Metric

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


@Metric.register("multilabel_average_precision")
class MultilabelAveragePrecision(Metric):
    """
    Average precision for multiclass multilabel classification. Average precision
    approximately equals area under the precision-recall curve.
    - Average precision: scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html
    - Precision/recall: this is multilabel metric, so all labels are considered predictions (with their
        corresponding confidences) not just the one with the highest confidence.

    Two modes of calculating AP are implemented,
        - a fast but less accurate implementation that bins threshold. Supports the frequent use of get_metrics
        - a more accurate implemenatition when get_metrics(reset=True)
    The fast one tends to underestimate AP.
    AP - Fast_AP < 10/number_of_bins
    """

    def __init__(self, bins=1000, recall_thr = 0.40) -> None:
        """Args:
            bins: number of threshold bins for the fast computation of AP
            recall_thr: compute AP (or AUC) for recall values [0:recall_thr]
        """
        self.recall_thr = recall_thr

        self.sigmoid = nn.Sigmoid()

        # stores data for the accurate calculation of AP
        self.predictions = np.array([])
        self.gold_labels = np.array([])

        # stored data for the fast calculation of AP
        self.bins = bins
        self.bin_size = 1.0/self.bins
        self.correct_counts = np.array([0] * self.bins)
        self.total_counts = np.array([0] * self.bins)

    def __call__(self,
                 predictions: torch.Tensor,
                 gold_labels: torch.Tensor):
        """
        Parameters
        ----------
        predictions : ``torch.Tensor``, required.
            A tensor of predictions of shape (batch_size, ..., num_classes).
        gold_labels : ``torch.Tensor``, required.
            A tensor of zeros and ones of shape (batch_size, ..., num_classes). It must be the same
            shape as the ``predictions`` tensor.
        """
        predictions = self.sigmoid(predictions)  # sigmoid to make sure all values are [0:1]

        # Get the data from the Variables to avoid GPU memory leak
        predictions, gold_labels = self.unwrap_to_tensors(predictions, gold_labels)

        # Sanity check
        if gold_labels.shape != predictions.shape:
            raise ConfigurationError("gold_labels must have the same shape of predictions. "
                                     "Found shapes {} and {}".format(gold_labels.shape, predictions.shape))

        pred = predictions.numpy().ravel()
        gold = gold_labels.numpy().ravel()

        # udpate data of accurate computation of AP
        self.predictions = np.append(self.predictions, pred)
        self.gold_labels = np.append(self.gold_labels, gold)

        # update data of fast computation of AP
        idx = (self.bins - 1) - np.minimum((pred/self.bin_size).astype(int), self.bins - 1)

        gold_uniq_idx, gold_idx_count = np.unique(idx[np.nonzero(gold)], return_counts=True)
        self.correct_counts[gold_uniq_idx] = np.add(self.correct_counts[gold_uniq_idx], gold_idx_count)

        uniq_idx, idx_count = np.unique(idx, return_counts=True)
        self.total_counts[uniq_idx] = np.add(self.total_counts[uniq_idx], idx_count)

    def _thresholded_average_precision_score(self, precision, recall):
        if len(precision) == 0:
            return 0, -1
        index = np.argmin(abs(recall - self.recall_thr))
        filtered_precision = precision[:index + 1]
        filtered_recall = recall[:index + 1]
        ap = np.sum(np.diff(np.insert(filtered_recall, 0, 0)) * filtered_precision)
        return ap, index  # index of the value with recall = self.recall_thr (useful for logging)

    def get_metric(self, reset: bool = False):
        """
        Returns average precision.

        If reset=False, returns the fast AP.
        If reset=True, returns accurate AP, logs difference ebtween accurate and fast AP and
                       logs a list of points on the precision-recall curve.

        """
        correct_cumsum = np.cumsum(self.correct_counts)
        precision = correct_cumsum  / np.cumsum(self.total_counts)
        recall = correct_cumsum  / correct_cumsum[-1]
        isfinite = np.isfinite(precision)
        precision = precision[isfinite]
        recall = recall[isfinite]
        ap, index = self._thresholded_average_precision_score(precision, recall)  # fast AP because of binned values
        if reset:
            fast_ap = ap
            precision, recall, thresholds = precision_recall_curve(self.gold_labels, self.predictions)

            # _thresholded_average_precision_score assumes precision is descending and recall is ascending
            precision = precision[::-1]
            recall = recall[::-1]
            thresholds = thresholds[::-1]
            ap, index = self._thresholded_average_precision_score(precision, recall)  # accurate AP because of using all values
            logger.info("Fast AP: %0.4f -- Accurate AP: %0.4f", fast_ap, ap)
            if index >= len(thresholds):
                logger.info("Index = %d but len(thresholds) = %d. Change index to point to the end of the list.",
                            index, len(thresholds))
                index = len(thresholds) - 1
            logger.info("at index %d/%d (top %%%0.4f) -- threshold: %0.4f",
                        index, len(self.gold_labels), 100.0 * index / len(self.gold_labels), thresholds[index])

            # only keep the top predictions then reverse again for printing (to draw the AUC curve)
            precision = precision[:index + 1][::-1]
            recall = recall[:index + 1][::-1]
            thresholds = thresholds[:index + 1][::-1]

            next_step = thresholds[0]
            step_size = 0.005
            max_skip = int(len(thresholds) / 500)
            skipped = 0
            logger.info("Precision-Recall curve ... ")
            logger.info("precision, recall, threshold")
            for p, r, t in [x for x in zip(precision, recall, thresholds)]:
                if t < next_step and skipped < max_skip:
                    skipped += 1
                    continue
                skipped = 0
                next_step += step_size
                # logger.info("%0.4f, %0.4f, %0.4f", p, r, t)
            self.reset()
        return ap

    @overrides
    def reset(self):
        self.predictions = np.array([])
        self.gold_labels = np.array([])

        self.correct_counts = np.array([0] * self.bins)
        self.total_counts = np.array([0] * self.bins)