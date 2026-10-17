# SPDX-License-Identifier: Apache-2.0
"""Portable contracts for C follow-up statistics, scope and integrity gates."""

from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest

import summarize_c_followup as summarize


def metric_row(seed: int, model: str, score: float) -> dict:
    classification = {"accuracy": score, "balanced_accuracy": score}
    return {"seed": seed, "model": model, "auroc": score, "ap": score,
            "default": classification, "validation_threshold": classification}


def completed_stages() -> list[dict]:
    return [{"stage": stage, "returncode": 0,
             "started_at": f"2026-10-16T10:{index * 2:02}:00+00:00",
             "ended_at": f"2026-10-16T10:{index * 2 + 1:02}:00+00:00"}
            for index, stage in enumerate(summarize.STAGES)]


def training_fixture() -> tuple[dict, dict, list[dict]]:
    epochs = [{"epoch": index, "global_step": 1735 * (index + 1),
               "metrics": {"val_loss": loss, "train_loss_epoch": .5 - index * .01}}
              for index, loss in enumerate((.45, .35, .40, .39, .41))]
    complete = {"status": "complete", "epochs": 5, "global_step": 8675,
                "best_validation_loss": .35, "best_checkpoint": "epoch=1-step=3470.ckpt"}
    config = {"seed": 43, "batch_size": 32, "trainer": {"max_epochs": 5}, "resume": False,
              "model": {"quantization_enabled": True}, "model_checkpoint": {"monitor": "val_loss", "mode": "min"}}
    return complete, config, epochs


class CFollowupContracts(unittest.TestCase):
    def setUp(self):
        self.rows = [metric_row(seed, model, .8 + (seed - 42) / 100)
                     for seed in summarize.SEEDS for model in summarize.MODELS]

    def test_sample_sd_uses_three_seeds(self):
        stats = summarize.aggregate(list(reversed(self.rows)))
        self.assertAlmostEqual(stats["c"]["metrics"]["auroc"]["mean"], .81)
        self.assertAlmostEqual(stats["c"]["metrics"]["auroc"]["sample_std"], .01)
        self.assertEqual(stats["c"]["n"], 3)

    def test_strict_pairing_excludes_unmatched_seed42(self):
        qap = next(row for row in self.rows if (row["seed"], row["model"]) == (42, "qap_b"))
        qap["auroc"] += .3
        all_seeds = summarize.qap_minus_c(self.rows, summarize.SEEDS)["metrics"]["auroc"]
        strict = summarize.qap_minus_c(self.rows, summarize.NEW_SEEDS)["metrics"]["auroc"]
        self.assertAlmostEqual(all_seeds["mean"], .1)
        self.assertAlmostEqual(all_seeds["sample_std"], math.sqrt(.03))
        self.assertEqual(strict, {"by_seed": {"43": 0., "44": 0.}, "mean": 0., "sample_std": 0.})

    def test_aggregate_rejects_missing_and_duplicate_rows(self):
        for rows in (self.rows[:-1], self.rows + [self.rows[0]]):
            with self.subTest(rows=len(rows)), self.assertRaises(ValueError):
                summarize.aggregate(rows)

    def test_aggregate_rejects_nonfinite_metrics(self):
        self.rows[0]["auroc"] = math.nan
        with self.assertRaises(ValueError):
            summarize.aggregate(self.rows)

    def test_stage_contract_requires_all_six_in_order(self):
        stages = completed_stages()
        self.assertEqual(len(summarize.validate_stages(stages)), 6)
        for invalid in (stages[:-1], list(reversed(stages)), stages + stages[:1]):
            with self.subTest(stages=len(invalid)), self.assertRaises(ValueError):
                summarize.validate_stages(invalid)

    def test_failed_or_overlapping_stage_is_rejected(self):
        stages = completed_stages()
        stages[3]["returncode"] = 1
        with self.assertRaises(ValueError):
            summarize.validate_stages(stages)
        stages = completed_stages()
        stages[1]["started_at"] = stages[0]["started_at"]
        with self.assertRaises(ValueError):
            summarize.validate_stages(stages)

    def test_budget_and_best_checkpoint_are_native(self):
        complete, config, epochs = training_fixture()
        summarize.validate_budget(complete, config, epochs, 43)
        for field, value in (("epochs", 4), ("global_step", 8674),
                             ("best_checkpoint", "epoch=4-step=8675.ckpt"), ("best_validation_loss", .41)):
            invalid = {**complete, field: value}
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize.validate_budget(invalid, config, epochs, 43)

    def test_training_config_and_epoch_metrics_are_checked(self):
        complete, config, epochs = training_fixture()
        for field, value in (("seed", 44), ("batch_size", 64), ("resume", True)):
            with self.subTest(field=field), self.assertRaises(ValueError):
                summarize.validate_budget(complete, {**config, field: value}, epochs, 43)
        epochs[2]["metrics"]["train_loss_epoch"] = math.inf
        with self.assertRaises(ValueError):
            summarize.validate_budget(complete, config, epochs, 43)

    def test_record_hash_rejects_corruption_without_publishing_paths(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "complete.json").write_text('{"status":"complete"}')
            digest = summarize.sha256(root / "complete.json")
            records = summarize.Records(root)
            records.bind_queue_hashes({"private/" + summarize.RAW_MARKER + "complete.json": digest})
            self.assertEqual(records.hashes, {"complete.json": digest})
            (root / "complete.json").write_text('{"status":"failed"}')
            with self.assertRaisesRegex(ValueError, "SHA256 mismatch"):
                records.read("complete.json")
            with self.assertRaisesRegex(ValueError, "Unsafe"):
                records.fingerprint("../outside.json")

    def test_previous_rows_keep_teammate_origin_and_distinct_sources(self):
        previous = json.loads(Path(__file__).with_name("seed-results.json").read_text())
        snapshot = deepcopy(previous)
        rows, runs = summarize.initial_records(previous)
        qap = next(row for row in rows if (row["seed"], row["model"]) == (42, "qap_b"))
        self.assertEqual(qap["origin"], "teammate checkpoint, full WSL replay")
        self.assertEqual(qap["source_id"], "historical_seed42")
        self.assertEqual(runs[0]["source_id"], "previous_seed_runs")
        self.assertEqual(previous, snapshot)
        previous["per_seed"][2]["origin"] = "local matched training"
        with self.assertRaises(ValueError):
            summarize.initial_records(previous)

    def test_published_summary_recomputes_and_links_to_exact_inputs(self):
        root = Path(__file__).parent
        result = json.loads((root / "c-seed-results.json").read_text())
        self.assertEqual(result["three_seed_aggregate"], summarize.aggregate(result["per_seed"]))
        for label, seeds in (("42_43_44", summarize.SEEDS), ("43_44", summarize.NEW_SEEDS)):
            self.assertEqual(result["qap_b_minus_c"][label], summarize.qap_minus_c(result["per_seed"], seeds))
        provenance = result["provenance"]
        self.assertEqual(provenance["previous_results_sha256"], summarize.sha256(root / "seed-results.json"))
        self.assertEqual(provenance["summary_script_sha256"], summarize.sha256(root / "summarize_c_followup.py"))
        self.assertEqual(provenance["summary_dependency_sha256"], summarize.sha256(root / "summarize_seeds.py"))
        self.assertTrue(result["sources"]["previous_seed_runs"]["dirty"])
        self.assertTrue(result["sources"]["c_followup"]["dirty"])
        self.assertNotEqual(result["sources"]["previous_seed_runs"]["runners_sha256"]["train_stage.py"],
                            result["sources"]["c_followup"]["runners_sha256"]["runners/train_stage.py"])
        text = json.dumps(result)
        self.assertNotIn("/Users/", text)
        self.assertNotIn("/root/", text)
        self.assertNotIn(".h5\"", text)


if __name__ == "__main__":
    unittest.main()
