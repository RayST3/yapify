import os
from pathlib import Path
import subprocess
import sys

from yapwrap import __version__


def test_core_imports_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", "-c", (
            "import yapwrap; "
            "from yapwrap.agents import BaseAgent, CallableAgent, AgnoAgent; "
            "from yapwrap.stt import BaseSTT; "
            "from yapwrap.tts import BaseTTS; "
            "from yapwrap.vad import BaseVAD; "
            "from yapwrap.transports import BaseTransport; "
            "print(yapwrap.__version__)"
        )],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == __version__
