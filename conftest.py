"""Repository-wide pytest import and optional-dependency setup."""

from pathlib import Path
import os
import sys


SRC_DIR = Path(__file__).resolve().parent / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))


def _add_sumo_to_test_path():
    candidates = []
    if os.environ.get("SUMO_HOME"):
        candidates.append(Path(os.environ["SUMO_HOME"]) / "bin")
    for variable in ("ProgramFiles(x86)", "ProgramW6432"):
        if os.environ.get(variable):
            candidates.append(
                Path(os.environ[variable]) / "Eclipse" / "Sumo" / "bin"
            )
    for candidate in candidates:
        executable = candidate / "sumo.exe"
        if executable.exists():
            os.environ["PATH"] = str(candidate) + os.pathsep + os.environ.get("PATH", "")
            os.environ.setdefault("SUMO_HOME", str(candidate.parent))
            return


_add_sumo_to_test_path()
