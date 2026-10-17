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
"""Reconstruct and activate isolated source trees for CPU contract tests."""

from contextlib import contextmanager
import importlib
import io
from pathlib import Path
import subprocess
import sys
import tarfile
from tempfile import TemporaryDirectory

BASE = "a2a4d441485efd8423c44b35d7448a30ccd44437"
PROJECT_PACKAGES = ("models", "ARES", "tasks", "datasets", "data_module", "util", "schedulers")


def reconstruct_source(repository: Path, destination: Path, configurable: bool = True) -> Path:
    """Never read or change the repository's working files."""
    destination.mkdir(parents=True, exist_ok=False)
    archive = subprocess.check_output(["git", "-C", str(repository), "archive", BASE])
    with tarfile.open(fileobj=io.BytesIO(archive)) as stream:
        stream.extractall(destination, filter="data")
    patches = ["source.patch"] + (["quantization-config.patch"] if configurable else [])
    for name in patches:
        subprocess.run(["git", "apply", str(Path(__file__).with_name(name))],
                       cwd=destination, check=True, capture_output=True)
    return destination


def _is_project_module(name: str) -> bool:
    return any(name == prefix or name.startswith(prefix + ".") for prefix in PROJECT_PACKAGES)


@contextmanager
def activated_source(source: Path):
    """Tests run serially: temporarily replace only project imports, then restore them."""
    previous_path = list(sys.path)
    saved = {name: module for name, module in sys.modules.items() if _is_project_module(name)}
    for name in saved:
        del sys.modules[name]
    sys.path.insert(0, str(source))
    importlib.invalidate_caches()
    try:
        yield
    finally:
        for name in list(sys.modules):
            if _is_project_module(name):
                del sys.modules[name]
        sys.modules.update(saved)
        sys.path[:] = previous_path
        importlib.invalidate_caches()


@contextmanager
def activated_precision_module():
    """Pure metadata tests need only the new configuration module, not a checkout."""
    with TemporaryDirectory() as temporary:
        root = Path(temporary)
        subprocess.run(["git", "apply", "--include=models/femba_quantization.py",
                        str(Path(__file__).with_name("quantization-config.patch"))],
                       cwd=root, check=True, capture_output=True)
        (root / "models/__init__.py").touch()
        with activated_source(root):
            yield
