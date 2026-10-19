import os
from pathlib import Path
import subprocess
import sys


def test_core_imports_without_site_packages():
    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        [sys.executable, "-S", "-c", (
            "import yapify; "
            "from yapify.agents import BaseAgent, CallableAgent, AgnoAgent; "
            "from yapify.stt import BaseSTT; "
            "from yapify.tts import BaseTTS; "
            "from yapify.vad import BaseVAD; "
            "from yapify.transports import BaseTransport; "
            "print(yapify.__version__)"
        )],
        env={**os.environ, "PYTHONPATH": str(root / "src")},
        capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "0.1.0"
