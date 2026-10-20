"""Test audio utilities in yapwrap."""

import struct
import pytest
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


@pytest.mark.parametrize("from_rate,to_rate", [(24000, 16000), (48000, 16000), (16000, 24000), (16000, 16000), (48000, 8000), (44100, 16000)])
@pytest.mark.parametrize("chunk_size", [1, 7, 640, 10000])
def test_streaming_resampler_is_independent_of_chunk_boundaries(from_rate, to_rate, chunk_size):
    from yapwrap.core.audio import PCM16Stream

    values = [(i * 137) % 40000 - 20000 for i in range(1001)]
    source = struct.pack(f"<{len(values)}h", *values)
    expected = []
    for index in range(len(values) * to_rate // from_rate):
        position, fraction = divmod(index * from_rate, to_rate)
        left, right = values[position], values[min(position + 1, len(values) - 1)]
        expected.append(int(left + (right - left) * fraction / to_rate))
    stream = PCM16Stream(from_rate, to_rate)
    chunks = [stream.process(source[i:i + chunk_size]) for i in range(0, len(source), chunk_size)]
    chunks.append(stream.finish())
    assert all(len(chunk) % 2 == 0 for chunk in chunks)
    assert b"".join(chunks) == struct.pack(f"<{len(expected)}h", *expected)


def test_streaming_pcm_rejects_truncated_sample_and_wav_uses_real_header():
    import io
    import wave
    from yapwrap.core.audio import PCM16Stream, wav_to_pcm16

    stream = PCM16Stream(16000, 16000)
    assert stream.process(b"\x01") == b""
    with pytest.raises(ValueError, match="incomplete PCM16"):
        stream.finish()
    output = io.BytesIO()
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x01\x02" * 10)
    content = output.getvalue()
    # A valid extra RIFF chunk makes the data header larger than 44 bytes.
    junk = b"JUNK" + struct.pack("<I", 4) + b"test"
    content = content[:4] + struct.pack("<I", len(content) + len(junk) - 8) + content[8:12] + junk + content[12:]
    assert wav_to_pcm16(content, 16000) == b"\x01\x02" * 10

