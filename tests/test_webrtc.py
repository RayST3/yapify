"""WebRTC framing and a real local aiortc exchange; no cloud calls."""

import asyncio
import json
from contextlib import aclosing
from fractions import Fraction

import pytest

pytest.importorskip("aiortc")
import numpy as np
from aiortc import AudioStreamTrack, RTCConfiguration, RTCPeerConnection
from aiortc.mediastreams import MediaStreamError
from av import AudioFrame

from yapwrap import SessionEvent
from yapwrap.transports import WebRTCTransport


async def test_input_converts_stereo_48k_to_mono_16k():
    class Track:
        count = 0

        async def recv(self):
            if self.count == 10:
                raise MediaStreamError
            indexes = np.arange(960) + self.count * 960
            tone = (np.sin(indexes * (2 * np.pi * 1000 / 48000)) * 1000).astype(np.int16)
            stereo = np.column_stack((tone, tone)).reshape(1, -1)
            frame = AudioFrame.from_ndarray(stereo, format="s16", layout="stereo")
            frame.sample_rate = 48000
            frame.pts = self.count * 960
            frame.time_base = Fraction(1, 48000)
            self.count += 1
            return frame

    transport = WebRTCTransport(track=Track())
    pcm = b"".join([chunk async for chunk in transport.audio_in()])
    samples = np.frombuffer(pcm, dtype="<i2")
    assert 3150 <= len(samples) <= 3200  # 200 ms, less the resampler's small delay.
    frequencies = np.fft.rfftfreq(len(samples), 1 / 16000)
    peak = frequencies[np.abs(np.fft.rfft(samples)).argmax()]
    assert peak == pytest.approx(1000, abs=10)


async def test_output_backpressure_releases_on_close():
    transport = WebRTCTransport()
    producer = asyncio.create_task(transport.send_audio(b"\x01\x00" * 32000))
    await asyncio.sleep(0)
    assert not producer.done()
    await transport.close()
    await asyncio.wait_for(producer, 1)


async def test_real_peer_audio_and_transcript_channel():
    client = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    server = RTCPeerConnection(RTCConfiguration(iceServers=[]))
    transport = WebRTCTransport(pc=server)
    received = asyncio.Queue()
    remote_track = asyncio.get_running_loop().create_future()
    channel_ready = asyncio.Event()
    channel = client.createDataChannel("events")
    channel.on("message", received.put_nowait)
    client.on("track", lambda track: remote_track.set_result(track))

    @server.on("datachannel")
    def on_channel(data_channel):
        transport.data_channel = data_channel
        channel_ready.set()

    client.addTrack(AudioStreamTrack())
    try:
        await client.setLocalDescription(await client.createOffer())
        await server.setRemoteDescription(client.localDescription)
        server.addTrack(transport.create_output_track())
        await transport.send_audio(b"\xe8\x03" * 16000)
        await server.setLocalDescription(await server.createAnswer())
        await client.setRemoteDescription(server.localDescription)
        track = await asyncio.wait_for(remote_track, 10)
        await asyncio.wait_for(channel_ready.wait(), 10)
        async with aclosing(transport.audio_in()) as incoming:
            pcm = await asyncio.wait_for(anext(incoming), 10)
            assert pcm and len(pcm) % 2 == 0
            audio = await asyncio.wait_for(track.recv(), 10)
            assert audio.sample_rate == 48000  # Opus clock at the receiving peer.
            assert np.any(audio.to_ndarray())
            event = SessionEvent("transcript", {"turn_id": 1, "text": "hello", "is_final": True})
            await transport.send_event(event)
            message = json.loads(await asyncio.wait_for(received.get(), 5))
            assert message["type"] == "transcript" and message["data"]["text"] == "hello"
    finally:
        await transport.close()
        await asyncio.gather(client.close(), server.close())
