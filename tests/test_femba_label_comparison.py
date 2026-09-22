"""Protect the paired label intervention and native per-batch metric replay."""
import copy
import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from scripts.femba_label_comparison import (
    _native_metrics,
    _normalized_config,
    _paired_labels,
    _label_digest,
    _require_exact_metrics,
    _require_same_recipe,
)


class LabelComparisonTests(unittest.TestCase):
    def test_paired_labels_accept_recorded_positive_to_negative_transitions(self):
        missing = [name for name in ("h5py", "numpy") if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest(f"Actual HDF5 dependencies unavailable: {', '.join(missing)}")
        import h5py
        import numpy as np

        old_y = np.array([0, 0, 1, 1])
        new_y = np.array([0, 1, 0, 1])
        with tempfile.TemporaryDirectory() as directory:
            old_path, new_path = Path(directory) / "old.h5", Path(directory) / "new.h5"
            with h5py.File(old_path, "w") as old:
                old.create_dataset("data_group_0/X", data=np.zeros((4, 2)))
                old.create_dataset("data_group_0/y", data=old_y)
            with h5py.File(new_path, "w") as new:
                group = new.create_group("data_group_0")
                group["X"] = h5py.ExternalLink("old.h5", "/data_group_0/X")
                group.create_dataset("y", data=new_y)
            output = {"old_labels_sha256": _label_digest(old_y), "new_labels_sha256": _label_digest(new_y),
                      "windows": 4, "positive": 2, "changed_labels": 2, "transition_matrix": [[1, 1], [1, 1]]}
            control = {"audit": {"test_windows": 4}}
            paper = {"audit": {"test_windows": 4}, "manifest": {"outputs": {"test": output}}}
            with patch("scripts.femba_label_comparison._paired_data_paths", return_value=(old_path, new_path)):
                original, relabeled = _paired_labels(control, paper)
                np.testing.assert_array_equal(original, old_y)
                np.testing.assert_array_equal(relabeled, new_y)
                output["transition_matrix"] = [[2, 0], [0, 2]]
                with self.assertRaisesRegex(ValueError, "transitions differ"):
                    _paired_labels(control, paper)

    def test_native_replay_preserves_per_batch_auto_sigmoid(self):
        missing = [name for name in ("torch", "torchmetrics")
                   if importlib.util.find_spec(name) is None]
        if missing:
            self.skipTest(f"Actual metric dependencies unavailable: {', '.join(missing)}")
        import torch

        logits = torch.tensor([[0., .6], [0., .8], [0., .9], [0., 2.]])
        labels = torch.tensor([0, 1, 0, 1])
        batched = _native_metrics(logits, labels, 2, "cpu")
        global_update = _native_metrics(logits, labels, 4, "cpu")
        self.assertEqual(batched["test_BinaryAUROC"], 1.0)
        self.assertEqual(global_update["test_BinaryAUROC"], 0.75)

    def test_reference_metric_drift_is_a_failure(self):
        expected = {"test_BinaryAUROC": .9238886833190918}
        _require_exact_metrics(expected, expected.copy())
        with self.assertRaisesRegex(ValueError, "exactly reproduce"):
            _require_exact_metrics({"test_BinaryAUROC": .9239889402101403}, expected)
        with self.assertRaisesRegex(ValueError, "keys differ"):
            _require_exact_metrics({}, expected)

    def test_only_data_and_output_paths_may_change(self):
        config = {
            "model": {"classification_type": "bc", "num_classes": 2},
            "data_module": {split: {"hdf5_file": f"/old/{split}.h5"}
                            for split in ("train", "val", "test")},
            "io": {"base_output_path": "/old/logs", "checkpoint_dirpath": "/old/checkpoints"},
            "optimizer": {"lr": 5e-4},
        }
        control = {key: {} for key in ("upstream_commit", "released_sha256", "source_hashes",
                                      "precision", "training_budget")}
        control.update({"config": config, "executed_adapter": {"sha256": "same-source"}})
        paper = copy.deepcopy(control)
        for split in ("train", "val", "test"):
            paper["config"]["data_module"][split]["hdf5_file"] = f"/paper/{split}.h5"
        paper["config"]["io"] = {"base_output_path": "/paper/logs", "checkpoint_dirpath": "/paper/checkpoints"}
        _require_same_recipe(control, paper)
        paper["config"]["optimizer"]["lr"] = 1e-4
        with self.assertRaisesRegex(ValueError, "configurations differ"):
            _require_same_recipe(control, paper)
        self.assertEqual(config["data_module"]["test"]["hdf5_file"], "/old/test.h5")
        self.assertEqual(_normalized_config(config)["optimizer"]["lr"], 5e-4)


if __name__ == "__main__":
    unittest.main()
