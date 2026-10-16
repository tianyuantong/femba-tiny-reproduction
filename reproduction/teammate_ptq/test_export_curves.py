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
"""Contracts for separating fit curves from repeated final validation events."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

from export_curves import export_curves, fit_epoch_series


def event(tag: str, step: int, value: float, wall_time: float) -> dict:
    return {"tag": tag, "step": step, "value": value, "wall_time": wall_time, "event": 0}


class FitCurveTest(unittest.TestCase):
    def test_first_epoch_value_keeps_raw_input_and_excludes_final_validation(self):
        rows = [event("epoch", 9, 0., 1.), event("val_loss", 9, .8, 2.),
                event("epoch", 19, 1., 3.), event("val_loss", 19, .6, 4.),
                event("val_loss", 19, .2, 5.), event("epoch", 20, 2., 6.),
                event("val_loss", 20, .1, 7.)]
        original = [dict(row) for row in rows]
        self.assertEqual(fit_epoch_series(rows, "val_loss", epochs=2), [(1, .8), (2, .6)])
        self.assertEqual(rows, original)

    def test_missing_epoch_tag_does_not_guess_fit_boundaries(self):
        with self.assertRaisesRegex(ValueError, "epoch scalar"):
            fit_epoch_series([event("val_loss", 9, .8, 2.)], "val_loss", epochs=1)

    @unittest.skipUnless(importlib.util.find_spec("tensorboard"), "requires TensorBoard")
    def test_real_tensorboard_export_preserves_duplicate_scalar_and_tensor_events(self):
        from tensorboard.compat.proto.event_pb2 import Event
        from tensorboard.compat.proto.summary_pb2 import Summary
        from tensorboard.summary.writer.event_file_writer import EventFileWriter
        from tensorboard.util.tensor_util import make_tensor_proto

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "events"
            writer = EventFileWriter(str(source))
            entries = [("epoch", 9, 0.), ("val_loss", 9, .8), ("val_loss", 9, .6),
                       ("epoch", 10, 1.), ("val_loss", 10, .2)]
            for ordinal, (tag, step, value) in enumerate(entries):
                writer.add_event(Event(wall_time=ordinal + 1., step=step,
                                       summary=Summary(value=[Summary.Value(tag=tag, simple_value=value)])))
            writer.add_event(Event(wall_time=7., step=9, summary=Summary(value=[
                Summary.Value(tag="val_BinaryAUROC", tensor=make_tensor_proto(.75))])))
            writer.close()
            manifest = root / "runs.json"
            manifest.write_text(json.dumps({"runs": [{"run": "fixture", "stage": "finetune",
                "seed": 42, "epochs": 1, "tensorboard_dir": str(source)}]}))
            result = export_curves(manifest, root / "output")
            self.assertEqual(result["scalar_events"], 6)
            self.assertEqual(len((root / "output/scalars.csv").read_text().splitlines()), 7)
            self.assertEqual(result["nonfinite_events"], 0)
            self.assertTrue((root / "output/native-positive-logit-auroc.svg").is_file())


if __name__ == "__main__":
    unittest.main()
