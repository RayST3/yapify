"""Framework-independent orchestration for one voice conversation."""

import asyncio
from collections import deque
from contextlib import aclosing
from typing import AsyncIterator, Deque, Optional, Set

from yapify.core.audio import AudioReslicer, DEFAULT_SAMPLE_RATE
from yapify.core.events import SessionEvent, TranscriptEvent
from yapify.core.memory import InMemoryHistory
from yapify.core.phrases import stream_phrases
from yapify.core.protocols import Agent, STT, Transport, TTS, VAD


class VoiceSession:
    """Link a transport, VAD, STT, agent and TTS for one connection.

    Input EOF means disconnect: pending work is cancelled and awaited.
    Provider failures propagate to the caller instead of leaving a silent call.
    Components must not be shared with another running session.
    """

    def __init__(
        self,
        transport: Transport,
        vad: VAD,
        stt: STT,
        agent: Agent,
        tts: TTS,
        memory: Optional[InMemoryHistory] = None,
        sample_rate: int = DEFAULT_SAMPLE_RATE,
        frame_bytes: Optional[int] = None,
        preroll_frames: int = 8,
    ) -> None:
        if sample_rate <= 0 or preroll_frames < 0:
            raise ValueError("sample_rate must be positive and preroll_frames nonnegative")
        self.transport = transport
        self.vad = vad
        self.stt = stt
        self.agent = agent
        self.tts = tts
        self.memory = memory if memory is not None else InMemoryHistory()
        self.sample_rate = sample_rate
        self.frame_bytes = frame_bytes if frame_bytes is not None else int(sample_rate * 0.02) * 2
        self.preroll_frames = preroll_frames
        for name, component in (("transport", transport), ("vad", vad), ("stt", stt), ("tts", tts)):
            rate = getattr(component, "sample_rate", sample_rate)
            if rate != sample_rate:
                raise ValueError(f"{name} sample_rate ({rate}) must match session sample_rate ({sample_rate})")

        self._reslicer = AudioReslicer(frame_bytes=self.frame_bytes)
        self._preroll: Deque[bytes] = deque(maxlen=preroll_frames)
        self._is_running = False
        self._is_speaking = False
        self._turn_id = 0
        self._stt_queue: Optional[asyncio.Queue] = None
        self._transcription_task: Optional[asyncio.Task] = None
        self._active_reply_task: Optional[asyncio.Task] = None
        self._tasks: Set[asyncio.Task] = set()
        self._stop_event = asyncio.Event()
        self._failure: Optional[asyncio.Future] = None

    def _spawn(self, coroutine) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error is not None and self._failure is not None and not self._failure.done():
                    self._failure.set_result(error)

        task.add_done_callback(finished)
        return task

    async def run(self) -> None:
        """Run until disconnect, stop(), cancellation, or a provider failure."""
        if self._is_running:
            raise RuntimeError("This VoiceSession is already running")
        self._is_running = True
        self._stop_event.clear()
        self._failure = asyncio.get_running_loop().create_future()
        input_task = asyncio.create_task(self._receive_audio())
        stop_task = asyncio.create_task(self._stop_event.wait())
        try:
            await asyncio.wait({input_task, stop_task, self._failure}, return_when=asyncio.FIRST_COMPLETED)
            if self._failure.done():
                raise self._failure.result()
            if input_task.done():
                await input_task
        finally:
            self._is_running = False
            tasks = {input_task, stop_task, *self._tasks}
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            self._failure.cancel()
            self._tasks.clear()
            self._stt_queue = None
            self._is_speaking = False
            self._reslicer.clear()
            self._preroll.clear()
            reset = getattr(self.vad, "reset", None)
            if reset is not None:
                reset()

    async def _receive_audio(self) -> None:
        stream = self.transport.audio_in()
        try:
            async for chunk in stream:
                for frame in self._reslicer.process(chunk):
                    await self._process_frame(frame)
                await asyncio.sleep(0)
        finally:
            close = getattr(stream, "aclose", None)
            if close is not None:
                await close()

    async def _emit(self, event_type: str, **data) -> None:
        sink = getattr(self.transport, "send_event", None)
        if sink is not None:
            await sink(SessionEvent(event_type, data))

    async def _process_frame(self, frame: bytes) -> None:
        event = self.vad.process(frame)
        if event == "start" and not self._is_speaking:
            self._is_speaking = True
            self._turn_id += 1
            # Cancel first so no stale audio can follow the stop_playback signal.
            pending = [task for task in (self._active_reply_task, self._transcription_task) if task is not None and not task.done()]
            for task in pending:
                task.cancel()
            try:
                await self.transport.stop_playback()
            finally:
                if pending:
                    await asyncio.gather(*pending, return_exceptions=True)
            await self._emit("speech_started", turn_id=self._turn_id)
            queue: asyncio.Queue = asyncio.Queue()
            for past in self._preroll:
                queue.put_nowait(past)
            self._preroll.clear()
            queue.put_nowait(frame)
            self._stt_queue = queue
            self._transcription_task = self._spawn(self._run_transcription(queue, self._turn_id))
        elif self._is_speaking:
            self._stt_queue.put_nowait(frame)
            if event == "stop":
                self._is_speaking = False
                self._stt_queue.put_nowait(None)
                self._stt_queue = None
                await self._emit("speech_stopped", turn_id=self._turn_id)
        else:
            self._preroll.append(frame)

    @staticmethod
    async def _cancel(*tasks: Optional[asyncio.Task]) -> None:
        pending = [task for task in tasks if task is not None and not task.done()]
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)

    async def _run_transcription(self, queue: asyncio.Queue, turn_id: int) -> None:
        # Capture this turn's queue; a later utterance must never replace it.
        async def audio_stream() -> AsyncIterator[bytes]:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        committed = []
        interim = ""
        async with aclosing(audio_stream()) as audio:
            stream = self.stt.transcribe(audio)
            try:
                async for transcript in stream:
                    if isinstance(transcript, TranscriptEvent):
                        if transcript.is_final:
                            if transcript.text.strip():
                                committed.append(transcript.text.strip())
                            interim = ""
                        else:
                            interim = transcript.text
                    else:
                        interim = transcript
                    text = " ".join([*committed, interim.strip()]).strip()
                    if self._is_running and turn_id == self._turn_id:
                        # A provider's final segment is not yet the full turn.
                        await self._emit("transcript", turn_id=turn_id, text=text, is_final=False)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        text = " ".join([*committed, interim.strip()]).strip()
        if self._is_running and turn_id == self._turn_id and not self._is_speaking:
            await self._emit("transcript", turn_id=turn_id, text=text, is_final=True)
            if text:
                self._active_reply_task = self._spawn(self._run_agent_reply(text, turn_id))

    async def _run_agent_reply(self, user_prompt: str, turn_id: int) -> None:
        tokens: asyncio.Queue = asyncio.Queue(maxsize=128)
        response = ""
        completed = False
        status = "error"

        async def generate() -> None:
            nonlocal response
            stream = self.agent.respond(user_prompt)
            try:
                async for event in stream:
                    if event.type == "text_delta" and event.content:
                        response += event.content
                        await self._emit("agent_delta", turn_id=turn_id, text=event.content)
                        await tokens.put(event.content)
                await tokens.put(None)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()

        async def token_stream() -> AsyncIterator[str]:
            while True:
                token = await tokens.get()
                if token is None:
                    break
                yield token

        async def speak() -> None:
            async with aclosing(token_stream()) as text:
                async with aclosing(stream_phrases(text)) as phrases:
                    async for phrase in phrases:
                        stream = self.tts.synthesize(phrase)
                        try:
                            async for chunk in stream:
                                await self.transport.send_audio(chunk)
                        finally:
                            close = getattr(stream, "aclose", None)
                            if close is not None:
                                await close()

        tasks = [asyncio.create_task(generate()), asyncio.create_task(speak())]
        try:
            await asyncio.gather(*tasks)
            completed = True
            status = "completed"
        except asyncio.CancelledError:
            status = "interrupted"
            raise
        finally:
            await self._cancel(*tasks)
            # Sending PCM does not prove the user heard it; no playback ACK exists.
            self.memory.add(user_prompt, response, was_interrupted=not completed)
            await self._emit("agent_complete", turn_id=turn_id, text=response, status=status)

    def stop(self) -> None:
        """Wake a session even if it is blocked waiting for microphone input."""
        self._stop_event.set()
