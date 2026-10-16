#*----------------------------------------------------------------------------*
#* Copyright (C) 2026.                                                        *
#* SPDX-License-Identifier: Apache-2.0                                        *
#*                                                                            *
#* Licensed under the Apache License, Version 2.0 (the "License");              *
#* you may not use this file except in compliance with the License.             *
#* You may obtain a copy of the License at                                      *
#*                                                                            *
#* http://www.apache.org/licenses/LICENSE-2.0                                   *
#*                                                                            *
#* Unless required by applicable law or agreed to in writing, software          *
#* distributed under the License is distributed on an "AS IS" BASIS,            *
#* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.    *
#* See the License for the specific language governing permissions and         *
#* limitations under the License.                                              *
#*----------------------------------------------------------------------------*
"""Small decision/ranking contracts; no model or GPU imports are required."""

import unittest

import numpy as np

from evaluate import decision_metrics, ranking_metrics


class MetricContractTest(unittest.TestCase):
    def test_common_logit_shift_preserves_classifier_ranking(self):
        labels = np.array([0, 0, 1, 1])
        logits = np.array([[1., 0.], [2., 0.], [0., 1.], [0., 2.]])
        shifted = logits + np.array([100., 200., -100., -200.])[:, None]
        self.assertEqual(decision_metrics(labels, logits), decision_metrics(labels, shifted))
        self.assertEqual(decision_metrics(labels, logits)["scores"]["logit_margin"]["auroc"], 1.)
        self.assertEqual(ranking_metrics(labels, shifted[:, 1])["auroc"], 0.)

    def test_positive_ranking_can_coexist_with_all_positive_decisions(self):
        result = decision_metrics(np.array([0, 0, 1, 1]),
                                  np.array([[0., 1.], [0., 2.], [0., 3.], [0., 4.]]))
        self.assertEqual(result["scores"]["logit_margin"]["auroc"], 1.)
        self.assertEqual(result["confusion_matrix"], [[0, 2], [0, 2]])
        self.assertEqual(result["balanced_accuracy"], 0.5)

    def test_ties_receive_half_auc_credit_and_select_class_zero(self):
        result = decision_metrics(np.array([0, 1]), np.array([[2., 2.], [-3., -3.]]))
        self.assertEqual(result["scores"]["logit_margin"]["auroc"], 0.5)
        self.assertEqual(result["scores"]["logit_margin"]["ap"], 0.5)
        self.assertEqual(result["confusion_matrix"], [[1, 0], [1, 0]])

    def test_invalid_metric_inputs_are_rejected(self):
        for labels, scores in [([], []), ([0, 0], [0., 1.]), ([0, 1], [0., np.nan])]:
            with self.subTest(labels=labels), self.assertRaises(ValueError):
                ranking_metrics(np.asarray(labels), np.asarray(scores))
        with self.assertRaises(ValueError):
            decision_metrics(np.array([0, 1]), np.zeros((2, 3)))


if __name__ == "__main__":
    unittest.main()
