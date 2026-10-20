"""Audio utilities for yapwrap: framing, resampling, and format conversions."""

import math
import struct
from typing import List

DEFAULT_SAMPLE_RATE = 16000
DEFAULT_FRAME_MS = 20
DEFAULT_FRAME_SAMPLES = int(DEFAULT_SAMPLE_RATE * (DEFAULT_FRAME_MS / 1000))  # 320 samples
DEFAULT_FRAME_BYTES = DEFAULT_FRAME_SAMPLES * 2  # 640 bytes (PCM16 mono)


class AudioReslicer:
    """Slices arbitrary incoming audio byte chunks into uniform frame sizes."""

    def __init__(self, frame_bytes: int = DEFAULT_FRAME_BYTES) -> None:
        if frame_bytes <= 0 or frame_bytes % 2:
            raise ValueError("frame_bytes must be a positive, even number for PCM16 audio")
        self.frame_bytes = frame_bytes
        self._buffer = bytearray()

    def process(self, chunk: bytes) -> List[bytes]:
        """Add new audio data and return any complete frames."""
        self._buffer.extend(chunk)
        frames: List[bytes] = []
        while len(self._buffer) >= self.frame_bytes:
            frame = bytes(self._buffer[: self.frame_bytes])
            del self._buffer[: self.frame_bytes]
            frames.append(frame)
        return frames

    def flush(self) -> bytes:
        """Return any remaining bytes in the buffer and clear it."""
        remaining = bytes(self._buffer)
        self._buffer.clear()
        return remaining

    def clear(self) -> None:
        """Clear internal buffer."""
        self._buffer.clear()


def calculate_rms(pcm: bytes) -> float:
    """Calculate the Root Mean Square (RMS) energy of 16-bit PCM mono audio."""
    if len(pcm) < 2:
        return 0.0
    num_samples = len(pcm) // 2
    sum_squares = 0.0
    for i in range(0, num_samples * 2, 2):
        val = struct.unpack_from("<h", pcm, i)[0]
        sum_squares += val * val
    return math.sqrt(sum_squares / num_samples)


def resample_pcm(pcm: bytes, from_rate: int, to_rate: int) -> bytes:
    """Resample 16-bit linear PCM mono audio between sample rates.

    Uses numpy for acceleration if available; otherwise falls back to pure-Python
    linear interpolation.
    """
    if from_rate <= 0 or to_rate <= 0:
        raise ValueError("Sample rates must be positive")
    if len(pcm) % 2:
        raise ValueError("PCM16 audio must contain complete two-byte samples")
    if from_rate == to_rate or not pcm:
        return pcm

    try:
        import numpy as np

        samples = np.frombuffer(pcm, dtype="<i2").astype(np.float32)
        duration = len(samples) / from_rate
        target_len = int(duration * to_rate)
        if target_len <= 0:
            return b""
        source_x = np.linspace(0, duration, len(samples), endpoint=False)
        target_x = np.linspace(0, duration, target_len, endpoint=False)
        resampled = np.interp(target_x, source_x, samples).clip(-32768, 32767).astype("<i2")
        return resampled.tobytes()
    except ImportError:
        pass

    # Pure Python linear interpolation fallback
    num_source = len(pcm) // 2
    if num_source == 0:
        return b""
    duration = num_source / from_rate
    num_target = int(duration * to_rate)
    if num_target <= 0:
        return b""

    ratio = from_rate / to_rate
    out = bytearray(num_target * 2)

    for i in range(num_target):
        src_pos = i * ratio
        idx = int(src_pos)
        frac = src_pos - idx

        if idx >= num_source - 1:
            val = struct.unpack_from("<h", pcm, (num_source - 1) * 2)[0]
        else:
            val1 = struct.unpack_from("<h", pcm, idx * 2)[0]
            val2 = struct.unpack_from("<h", pcm, (idx + 1) * 2)[0]
            val = int(val1 + frac * (val2 - val1))

        val = max(-32768, min(32767, val))
        struct.pack_into("<h", out, i * 2, val)

    return bytes(out)


def mulaw_to_pcm16(data: bytes) -> bytes:
    """Decode 8-bit G.711 mu-law audio to 16-bit PCM."""
    table = _get_mulaw_to_pcm_table()
    out = bytearray(len(data) * 2)
    for i, b in enumerate(data):
        val = table[b]
        struct.pack_into("<h", out, i * 2, val)
    return bytes(out)


def pcm16_to_mulaw(pcm: bytes) -> bytes:
    """Encode 16-bit PCM audio to 8-bit G.711 mu-law."""
    num_samples = len(pcm) // 2
    out = bytearray(num_samples)
    for i in range(num_samples):
        sample = struct.unpack_from("<h", pcm, i * 2)[0]
        out[i] = _linear_to_mulaw(sample)
    return bytes(out)


_MULAW_TO_PCM: List[int] = []


def _get_mulaw_to_pcm_table() -> List[int]:
    global _MULAW_TO_PCM
    if not _MULAW_TO_PCM:
        table = []
        for i in range(256):
            b = ~i & 0xFF
            sign = (b & 0x80) >> 7
            exponent = (b & 0x70) >> 4
            mantissa = b & 0x0F
            sample = ((mantissa << 3) + 132) << exponent
            sample -= 132
            table.append(-sample if sign else sample)
        _MULAW_TO_PCM = table
    return _MULAW_TO_PCM


def _linear_to_mulaw(sample: int) -> int:
    BIAS = 132
    CLIP = 32635
    sign = (sample >> 8) & 0x80
    if sign != 0:
        sample = -sample
    if sample > CLIP:
        sample = CLIP
    sample += BIAS
    exponent = 7
    for exp in range(7, -1, -1):
        if sample & (1 << (exp + 7)):
            exponent = exp
            break
    mantissa = (sample >> (exponent + 3)) & 0x0F
    return ~(sign | (exponent << 4) | mantissa) & 0xFF

