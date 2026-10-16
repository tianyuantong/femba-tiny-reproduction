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
"""Deterministic contracts for validation-only threshold control."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import threshold_control as control


def create_captures(root: Path, test_scores=None) -> None:
    labels = np.array([0, 1, 0, 1, 0, 0], dtype=np.int64)
    val_scores = np.array([0.2, 0.8, 0.4, 0.9, 0.4, 0.1])
    if test_scores is None:
        test_scores = val_scores[::-1].copy()
    for model_id in control.MODEL_IDS:
        for split, scores in (("val", val_scores), ("test", test_scores)):
            array = root / f"{model_id}_{split}_logits.npz"
            np.savez_compressed(array, labels=labels, logits=np.column_stack((np.zeros(6), scores)))
            metadata = {"model": model_id, "split": split, "windows": 6,
                        "array_sha256": control.digest(array), "checkpoint_sha256": model_id,
                        "labels_sha256": hashlib.sha256(labels.tobytes()).hexdigest()}
            (root / f"{model_id}_{split}.json").write_text(json.dumps(metadata))


class ThresholdContracts(unittest.TestCase):
    def test_tied_ranking(self):
        metrics = control.ranking_metrics(np.array([0, 1, 0, 1]), np.array([0.0, 0.5, 0.5, 1.0]))
        self.assertEqual(metrics["auroc"], 0.875)
        self.assertAlmostEqual(metrics["ap"], 5 / 6)

    def test_unbalanced_objective_and_threshold_ties(self):
        labels = np.array([0, 0, 0, 1])
        scores = np.array([-2.0, -1.0, 0.5, 1.0])
        selected = control.select_threshold(labels, scores)
        self.assertEqual(selected["threshold"], 0.5)
        self.assertEqual(selected["validation_selected"]["balanced_accuracy"], 1.0)
        tied = control.select_threshold(np.array([0, 1]), np.array([1.0, 1.0]))
        self.assertEqual(tied["threshold"], 0.0)
        self.assertEqual(tied["validation_selected"]["confusion_matrix"], [[0, 1], [0, 1]])
        # Equal best accuracy and absolute distance use the lower threshold.
        tied = control.select_threshold(np.array([0, 0, 0, 1, 1, 1]),
                                        np.array([-1.0, 1.0, 2.0, 0.0, 2.0, 3.0]))
        self.assertEqual(tied["threshold"], -1.0)

    def test_matches_brute_force_partitions(self):
        generator = np.random.default_rng(42)
        for _ in range(50):
            labels = np.r_[0, 1, generator.integers(0, 2, 18)]
            scores = generator.integers(-3, 4, 20).astype(np.float64)
            thresholds = np.unique(np.r_[scores, 0.0, np.nextafter(scores.min(), -np.inf)])
            candidates = [(control.binary_metrics(labels, scores, value)["balanced_accuracy"], value)
                          for value in thresholds]
            best_accuracy = max(item[0] for item in candidates)
            best = min((value for accuracy, value in candidates if np.isclose(accuracy, best_accuracy)),
                       key=lambda value: (abs(value), value))
            self.assertEqual(control.select_threshold(labels, scores)["threshold"], best)

    def test_invalid_capture_domain(self):
        for labels, scores in ((np.array([0, 0]), np.array([1.0, 2.0])),
                               (np.array([0, 1]), np.array([np.nan, 2.0])),
                               (np.array([0.0, 1.0]), np.array([1.0, 2.0]))):
            with self.assertRaises(ValueError):
                control.select_threshold(labels, scores)

    def test_selection_precedes_any_test_read_and_is_test_independent(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = root / "first", root / "second"
            first.mkdir()
            second.mkdir()
            create_captures(first)
            create_captures(second, np.array([99.0, -3.0, 8.0, 10.0, -4.0, 1.0]))
            original_load = control.load_capture
            current_output = [root / "result_a"]

            def guarded_load(captures, model_id, split):
                if split == "test":
                    selection = json.loads((current_output[0] / "selection.json").read_text())
                    self.assertEqual(len(selection["models"]), 4)
                    self.assertTrue((current_output[0] / "test_started.json").exists())
                return original_load(captures, model_id, split)

            with patch.object(control, "load_capture", side_effect=guarded_load):
                result_a = control.run(first, current_output[0])
                current_output[0] = root / "result_b"
                result_b = control.run(second, current_output[0])
            selection_a = json.loads((root / "result_a/selection.json").read_text())
            selection_b = json.loads((root / "result_b/selection.json").read_text())
            self.assertEqual(selection_a["models"], selection_b["models"])
            self.assertNotEqual(result_a["models"], result_b["models"])
            self.assertTrue(result_a["captures_unchanged"])
            with self.assertRaises(FileExistsError):
                control.run(first, root / "result_a")

    def test_source_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            captures = root / "captures"
            captures.mkdir()
            create_captures(captures)
            original_load = control.load_capture

            def mutating_load(captures, model_id, split):
                value = original_load(captures, model_id, split)
                if split == "test" and model_id == control.MODEL_IDS[0]:
                    path = captures / "fp32_val.json"
                    path.write_text(path.read_text() + " ")
                return value

            with patch.object(control, "load_capture", side_effect=mutating_load):
                with self.assertRaises(AssertionError):
                    control.run(captures, root / "result")
            self.assertFalse((root / "result/results.json").exists())


if __name__ == "__main__":
    unittest.main()
