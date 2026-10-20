import struct
from types import SimpleNamespace

import pytest

from yapwrap.vad.silero import SileroVAD


def test_silero_reframes_without_losing_or_repeating_samples():
    np = pytest.importorskip("numpy")
    windows = []
    vad = SileroVAD()
    vad._np = np
    vad._torch = SimpleNamespace(from_numpy=lambda samples: samples)

    def infer(samples):
        assert len(samples) == 512
        windows.append(samples.copy())
        return {"start": 0} if len(windows) == 1 else {"end": 512}

    vad._iterator = infer
    pcm = struct.pack("<" + "h" * 1280, *range(1280))
    events = [vad.process(pcm[i:i + 640]) for i in range(0, len(pcm), 640)]
    assert events == [None, "start", None, "stop"]
    assert np.array_equal(np.concatenate(windows) * 32768, np.arange(1024))
    vad._iterator = SimpleNamespace(reset_states=lambda: None)
    vad.reset()
    assert vad._reslicer.flush() == b""
    assert not vad._events


def test_installed_silero_accepts_twenty_ms_frames():
    pytest.importorskip("silero_vad")
    vad = SileroVAD()
    assert [vad.process(b"\x00" * 640) for _ in range(50)] == [None] * 50
    vad.reset()

