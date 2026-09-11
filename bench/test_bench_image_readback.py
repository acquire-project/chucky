"""Exercise the independent verifier through its public command line."""

from pathlib import Path
import subprocess
import sys
import tempfile

executable, backend = sys.argv[1:]
verifier = Path(__file__).resolve().parents[1] / "scripts/datasets/verify_output.py"
with tempfile.TemporaryDirectory(prefix="chucky-image-readback-") as directory:
    subprocess.run(
        ["uv", "run", str(verifier), "--executable", executable,
         "--backends", backend, "--output", str(Path(directory) / "checks")],
        check=True,
    )
