"""Test audio utilities in yapwrap."""

import struct
from yapwrap.core.audio import (
    AudioReslicer,
    calculate_rms,
    resample_pcm,
    mulaw_to_pcm16,
    pcm16_to_mulaw,
)


def test_audio_reslicer():
    # Target frame size = 640 bytes (320 samples @ 16kHz)
    reslicer = AudioReslicer(frame_bytes=640)

    # Feed 200 bytes
    frames = reslicer.process(b"\x00" * 200)
    assert len(frames) == 0

    # Feed 500 bytes -> total 700 bytes -> 1 frame of 640 bytes, 60 bytes left
    frames = reslicer.process(b"\x01" * 500)
    assert len(frames) == 1
    assert len(frames[0]) == 640

    # Feed 1280 bytes (2 full frames)
    frames = reslicer.process(b"\x02" * 1280)
    assert len(frames) == 2
    for f in frames:
        assert len(f) == 640

    # Flush remainder
    remaining = reslicer.flush()
    assert len(remaining) == 60


def test_calculate_rms():
    # Perfect silence
    silence = b"\x00\x00" * 320
    assert calculate_rms(silence) == 0.0

    # DC offset signal with constant value 1000
    signal = struct.pack("<h", 1000) * 320
    rms = calculate_rms(signal)
    assert abs(rms - 1000.0) < 0.1


def test_resample_pcm():
    # 1 second of 16kHz audio = 32000 bytes (16000 samples)
    source_16k = b"\x00\x05" * 16000
    assert len(source_16k) == 32000

    # Resample 16kHz -> 24kHz (should be 24000 samples = 48000 bytes)
    resampled_24k = resample_pcm(source_16k, from_rate=16000, to_rate=24000)
    assert len(resampled_24k) == 48000

    # Resample 24kHz -> 16kHz (back to 16000 samples = 32000 bytes)
    resampled_16k = resample_pcm(resampled_24k, from_rate=24000, to_rate=16000)
    assert len(resampled_16k) == 32000

    # Same rate returns exact input
    assert resample_pcm(source_16k, 16000, 16000) == source_16k


def test_mulaw_transcoding_roundtrip():
    # Generate 16-bit PCM samples
    pcm_in = bytearray()
    for val in [-10000, -500, 0, 500, 10000]:
        pcm_in.extend(struct.pack("<h", val))

    # Encode to mu-law (8-bit)
    mulaw = pcm16_to_mulaw(bytes(pcm_in))
    assert len(mulaw) == len(pcm_in) // 2

    # Decode back to PCM16
    pcm_out = mulaw_to_pcm16(mulaw)
    assert len(pcm_out) == len(pcm_in)

    # Mu-law is lossy, but relative values should match closely
    for i in range(0, len(pcm_in), 2):
        val_in = struct.unpack_from("<h", pcm_in, i)[0]
        val_out = struct.unpack_from("<h", pcm_out, i)[0]
        if val_in == 0:
            assert abs(val_out) <= 10
        else:
            rel_err = abs(val_in - val_out) / abs(val_in)
            assert rel_err < 0.15  # within mu-law quantization error

