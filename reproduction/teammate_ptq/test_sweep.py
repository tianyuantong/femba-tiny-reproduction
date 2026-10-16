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
"""Preset selection and validation/test separation contracts for the PTQ sweep."""

import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

import sweep


def validation_rows() -> list[dict]:
    return [{"candidate": name, "percentile": percentile, "split": "val",
             "auroc": 0.8, "ap": 0.7, "loss": 0.6,
             "threshold_selection": {"threshold": 0.25}}
            for name, percentile in sweep.CANDIDATES]


class SweepContracts(unittest.TestCase):
    def test_existing_checkpoint_paths_are_all_or_none(self):
        self.assertEqual(sweep.existing_checkpoints(SimpleNamespace()), {})
        paths = {name: Path(name + ".ckpt") for name in sweep.REUSE_OPTIONS}
        checkpoints = sweep.existing_checkpoints(SimpleNamespace(**paths))
        self.assertEqual(list(checkpoints), [name for name, _ in sweep.CANDIDATES[:4]])
        self.assertEqual(list(checkpoints.values()), list(paths.values()))
        for missing in sweep.REUSE_OPTIONS:
            incomplete = {name: path for name, path in paths.items() if name != missing}
            with self.assertRaises(ValueError):
                sweep.existing_checkpoints(SimpleNamespace(**incomplete))

    def test_candidate_and_metric_order_are_fixed(self):
        rows = validation_rows()
        self.assertEqual(sweep.choose_candidate(rows)["candidate"], "original")
        rows[6]["loss"] = 0.5
        self.assertEqual(sweep.choose_candidate(rows)["candidate"], "p90")
        rows[5]["ap"] = 0.71
        self.assertEqual(sweep.choose_candidate(rows)["candidate"], "p95")
        rows[4]["auroc"] = 0.81
        self.assertEqual(sweep.choose_candidate(rows)["candidate"], "p98")
        rows[4]["split"] = "test"
        with self.assertRaises(ValueError):
            sweep.choose_candidate(rows)
        with self.assertRaises(ValueError):
            sweep.choose_candidate(rows[:4])

    def test_margin_metrics_ignore_common_logit_shift(self):
        labels = np.array([0, 0, 1, 1])
        logits = np.array([[1., 0.], [2., 0.], [0., 1.], [0., 2.]])
        shifted = logits + np.array([100., 200., -100., -200.])[:, None]
        self.assertEqual(sweep.validation_metrics(labels, logits),
                         sweep.validation_metrics(labels, shifted))
        self.assertEqual(sweep.validation_metrics(labels, logits)["auroc"], 1.0)

    def test_selection_is_saved_before_test_and_independent_of_test(self):
        with tempfile.TemporaryDirectory() as temporary:
            outputs = [Path(temporary) / name for name in ("first", "second")]
            for index, output in enumerate(outputs):
                output.mkdir()

                def guarded_test(selected):
                    selection_path = output / "selection.json"
                    self.assertEqual(json.loads(selection_path.read_text())["selected"], selected)
                    started = json.loads((output / "test_started.json").read_text())
                    self.assertEqual(started["selection_sha256"], sweep.capture.sha256(selection_path))
                    return {"auroc": 0.1 + index}

                sweep.finish(validation_rows(), output, guarded_test)
            selections = [json.loads((output / "selection.json").read_text()) for output in outputs]
            self.assertEqual(selections[0]["selected"], selections[1]["selected"])
            self.assertEqual(selections[0]["validation_candidates"], selections[1]["validation_candidates"])
            with self.assertRaises(FileExistsError):
                sweep.finish(validation_rows(), outputs[0], lambda selected: self.fail("test repeated"))

    def test_selection_mutation_is_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)

            def corrupt_selection(selected):
                path = output / "selection.json"
                path.write_text(path.read_text() + " ")
                return {"auroc": 0.8}

            with self.assertRaises(AssertionError):
                sweep.finish(validation_rows(), output, corrupt_selection)
            self.assertFalse((output / "results.json").exists())


if __name__ == "__main__":
    unittest.main()
