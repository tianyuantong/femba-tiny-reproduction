"""Contracts for the label-only FP32 control; no CUDA or model required."""
import importlib.util
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np

from scripts.femba_paper_labels_prepare import (
    PROCESSOR_SHA256, UPSTREAM_COMMIT, _add_signal, _consume_signal,
    file_sha256, labels_from_csv, prepare, signal_sha256,
)


class LabelRuleTests(unittest.TestCase):
    def test_threshold_crossing_and_class_vocabulary_are_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "annotations.csv"
            csv.write_text("channel,start_time,stop_time,label\n"
                           "FP1-F7,0,1.5,musc\n"       # Exactly 30%.
                           "FP1-F7,9.99609375,10.00390625,eyem\n"  # Crosses windows.
                           "FP1-F7,15,16.5,cpsz\n"     # Parent-only class.
                           "INVALID,5,10,tcsz\n")
            old, new = labels_from_csv(csv, 4)
            self.assertEqual(old.tolist(), [1, 0, 0, 1])
            self.assertEqual(new.tolist(), [1, 1, 1, 0])

    def test_annotation_uses_python_ties_to_even_rounding(self):
        with tempfile.TemporaryDirectory() as directory:
            csv = Path(directory) / "annotations.csv"
            csv.write_text("channel,start_time,stop_time,label\n"
                           f"FP1-F7,0,{0.5 / 256},musc\n")
            _, new = labels_from_csv(csv, 1)
            self.assertEqual(new.tolist(), [0])

    def test_duplicate_signal_with_ambiguous_labels_is_rejected(self):
        index = {}
        _add_signal(index, "digest", 0, 0, "window-a")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _add_signal(index, "digest", 0, 1, "window-b")
        with self.assertRaisesRegex(ValueError, "ambiguous"):
            _add_signal(index, "digest", 1, 0, "window-c")

    def test_identical_duplicate_occurrences_are_consumed_exactly_once(self):
        signal = np.zeros((22, 1280), dtype=np.float64)
        digest = signal_sha256(signal)
        index = {}
        for identity in ("a", "b"):
            _add_signal(index, digest, 0, 1, identity)
        matched = {_consume_signal(index, signal, 0)[1] for _ in range(2)}
        self.assertEqual(matched, {"a", "b"})
        self.assertEqual(index[digest].source_ids, [])
        with self.assertRaisesRegex(ValueError, "unused matching"):
            _consume_signal(index, signal, 0)

    def test_equal_signal_with_wrong_original_label_is_rejected(self):
        signal = np.zeros((22, 1280), dtype=np.float64)
        index = {}
        _add_signal(index, signal_sha256(signal), 0, 1, "a")
        with self.assertRaisesRegex(ValueError, "label disagrees"):
            _consume_signal(index, signal, 1)


@unittest.skipUnless(importlib.util.find_spec("h5py"), "h5py required for full HDF5 contracts")
class PreparationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.parent_path, self.processed = self._fixture()

    def _fixture(self):
        import h5py

        processed, records, outputs = self.root / "processed", [], {}
        for split in ("train", "val", "test"):
            source = self.root / f"{split}_record.edf"
            source.touch()
            source.with_suffix(".csv").write_text("channel,start_time,stop_time,label\n"
                                                  "FP1-F7,0,1.5,musc\nFP1-F7,5,5.1,musc\n")
            records.append({"source": str(source), "split": split})
            folder = processed / split
            folder.mkdir(parents=True)
            signals = [np.full((22, 1280), i, dtype=np.float64) for i in range(3)]
            for index, signal in enumerate(signals):
                with (folder / f"upstream2025_{source.stem}_{index}.pkl").open("wb") as handle:
                    pickle.dump({"X": signal, "y": int(index == 0)}, handle)
            h5path = self.root / f"parent_{split}.h5"
            with h5py.File(h5path, "w") as handle:
                # Deliberately different from records, pickle order and natural group order.
                for group_name, index in (("data_group_2", 1), ("data_group_10", 2), ("data_group_0", 0)):
                    group = handle.create_group(group_name)
                    group.create_dataset("X", data=signals[index][None])
                    group.create_dataset("y", data=[int(index == 0)])
            outputs[split] = {"windows": 3, "positive": 1, "path": str(h5path)}
        manifest = {"status": "complete", "upstream_commit": UPSTREAM_COMMIT,
                    "source_sha256": PROCESSOR_SHA256, "artifact_fraction_threshold": 0.3,
                    "seed": 42, "split_unit": "record", "records": records, "outputs": outputs}
        parent_path = self.root / "parent-manifest.json"
        parent_path.write_text(json.dumps(manifest))
        return parent_path, processed

    def test_hash_mapping_preserves_loader_order_and_external_signals(self):
        import h5py

        before = file_sha256(self.parent_path)
        output = self.root / "control"
        result = prepare(self.parent_path, self.processed, output)
        self.assertEqual(result["parent_manifest"]["sha256"], before)
        self.assertEqual(file_sha256(self.parent_path), before)
        self.assertEqual(result["artifact_fraction_comparison"], ">")
        for split, metadata in result["outputs"].items():
            self.assertEqual(metadata["windows"], 3)
            self.assertEqual(metadata["positive"], 2)
            self.assertEqual(metadata["transition_matrix"], [[1, 1], [0, 1]])
            with h5py.File(metadata["source_h5"]["path"], "r") as parent, h5py.File(metadata["path"], "r") as new:
                self.assertEqual(list(new.keys()), ["data_group_0", "data_group_10", "data_group_2"])
                self.assertEqual(list(new.keys()), list(parent.keys()))
                self.assertEqual([int(new[name]["y"][0]) for name in new], [1, 0, 1])
                for name in parent:
                    self.assertIsInstance(new[name].get("X", getlink=True), h5py.ExternalLink)
                    np.testing.assert_array_equal(new[name]["X"][:], parent[name]["X"][:])
            self.assertEqual(file_sha256(Path(metadata["path"])), metadata["sha256"])
            self.assertEqual(file_sha256(output / metadata["mapping_file"]), metadata["mapping_sha256"])
        with self.assertRaises(FileExistsError):
            prepare(self.parent_path, self.processed, output)

    def test_parent_label_mismatch_cannot_publish_complete_manifest(self):
        import h5py

        with h5py.File(self.root / "parent_train.h5", "r+") as parent:
            parent["data_group_0/y"][0] = 0
        output = self.root / "control"
        with self.assertRaisesRegex(ValueError, "label disagrees"):
            prepare(self.parent_path, self.processed, output)
        self.assertFalse((output / "results/upstream2025-data.json").exists())
        self.assertTrue((output / "results/preparation-failed.json").exists())

    def test_changed_hdf5_signal_cannot_publish_complete_manifest(self):
        import h5py

        with h5py.File(self.root / "parent_train.h5", "r+") as parent:
            parent["data_group_0/X"][0, 0, 0] += 1
        output = self.root / "control"
        with self.assertRaisesRegex(ValueError, "unused matching"):
            prepare(self.parent_path, self.processed, output)
        self.assertFalse((output / "results/upstream2025-data.json").exists())


if __name__ == "__main__":
    unittest.main()
