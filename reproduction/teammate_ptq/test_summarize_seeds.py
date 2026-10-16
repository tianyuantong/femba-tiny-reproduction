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
"""Contracts for sample statistics and preserving the historical checkpoint origin."""

import math
from types import SimpleNamespace
import unittest

import summarize_seeds as summarize


def row(seed: int, model: str, score: float) -> dict:
    classification = {"accuracy": score, "balanced_accuracy": score}
    return {"seed": seed, "model": model, "auroc": score, "ap": score,
            "default": classification, "validation_threshold": classification}


class SeedSummaryContracts(unittest.TestCase):
    def test_sample_standard_deviation_and_seed_matched_differences(self):
        rows = [row(seed, model, seed / 100 + offset)
                for model, offset in (("qap_b", .1), ("fp32", 0), ("ptq", -.2))
                for seed in (44, 42, 43)]
        aggregate = summarize.aggregate(rows, [42, 43, 44])
        self.assertAlmostEqual(aggregate["fp32"]["metrics"]["auroc"]["sample_std"], .01)
        difference = summarize.differences(rows, [42, 43, 44])["ptq_minus_fp32"]
        self.assertAlmostEqual(difference["metrics"]["auroc"]["mean"], -.2)
        self.assertEqual(set(difference["metrics"]["auroc"]["by_seed"]), {"42", "43", "44"})
        self.assertAlmostEqual(difference["metrics"]["auroc"]["sample_std"], 0)
        with self.assertRaises(ValueError):
            summarize.summary([1, math.nan])

    def test_historical_ptq_uses_validation_selected_extended_sweep(self):
        history = {"results": [], "expanded_calibration": {
            "test": {"auroc": .8, "ap": .9, "checkpoint_sha256": "selected"},
            "selected_percentile": 95, "default": {}, "validation_selected": {}}}
        for model in ("fp32", "ptq_p99", "qap_b"):
            history["results"].append({"model": model, "AUROC": .7, "AP": .6,
                                       "checkpoint_sha256": model, "default": {},
                                       "test_with_validation_threshold": {}})
        rows = summarize.historical_rows(history)
        ptq = next(record for record in rows if record["model"] == "ptq")
        qap = next(record for record in rows if record["model"] == "qap_b")
        self.assertEqual(ptq["checkpoint_sha256"], "selected")
        self.assertEqual(ptq["selected_percentile"], 95)
        self.assertEqual(qap["origin"], "teammate checkpoint replay")

    def test_closeout_excludes_process_commands_and_rejects_active_training(self):
        captures = [{"metadata": f"capture{i}.json", "array_sha256": str(i),
                     "windows": 7211, "auroc": .9, "ap": .8} for i in range(26)]
        originals = {row["metadata"]: {**row, "scores": {"logit_margin": row}}
                     for row in captures}
        originals.update({
            "independent-final-metrics.json": {"result": "PASS", "library": "sklearn.metrics",
                                                "captures": 26, "records": captures, "checks": []},
            "seed-stages.jsonl": [{"returncode": 0}] * 25,
            "seed-status.json": {"pid": 193435},
            "live-closeout.json": {"time": "now", "gpu": "0 %, 53 MiB, 15998 MiB",
                                    "processes": ["PID PPID ELAPSED COMMAND",
                                                  "100 1 00:10 python independent_metrics.py --token SECRET"]}})
        records = SimpleNamespace(read=lambda name, **kwargs: originals[name])
        runs = [{"stage": "pretrain", "epochs": 10, "curve_scalar_events": 2883}] * 4
        runs += [{"stage": "finetune", "epochs": 5, "curve_scalar_events": 16155}] * 5
        result = summarize.verification_record(records, runs)
        self.assertTrue(result["closeout"]["training_and_evaluation_exited"])
        self.assertNotIn("SECRET", str(result))
        originals["live-closeout.json"]["processes"][1] = "101 1 00:10 python train_stage.py"
        with self.assertRaises(AssertionError):
            summarize.verification_record(records, runs)


if __name__ == "__main__":
    unittest.main()
