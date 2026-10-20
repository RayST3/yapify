"""Framework-independent orchestration for one voice conversation."""

import asyncio
from collections import deque
from collections.abc import AsyncGenerator
from contextlib import aclosing
from typing import Deque, Optional, Set

from yapwrap.core.audio import AudioReslicer, DEFAULT_SAMPLE_RATE
from yapwrap.core.events import SessionEvent, TranscriptEvent
from yapwrap.core.memory import InMemoryHistory
from yapwrap.core.base import BaseAgent, BaseSTT, BaseTTS, BaseVAD, BaseTransport


class YapSesh:
    """Link a transport, VAD, STT, agent and TTS for one connection.

    Input EOF means disconnect: pending work is cancelled and awaited.
    Provider failures propagate to the caller instead of leaving a silent call.
    Components must not be shared with another running session.
    """

    def __init__(
        self,
        transport: BaseTransport,
        vad: BaseVAD,
        stt: BaseSTT,
        agent: BaseAgent,
        tts: BaseTTS,
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
        self._cancelling: Set[asyncio.Task] = set()
        # Cleanup can outlive a turn. Keep each stateful component exclusive
        # without making microphone capture wait for the previous turn.
        self._stt_lock = asyncio.Lock()
        self._agent_lock = asyncio.Lock()
        self._tts_lock = asyncio.Lock()
        self._stop_event = asyncio.Event()
        self._failure: Optional[asyncio.Future] = None

    def _spawn(self, coroutine, turn_id: int) -> asyncio.Task:
        task = asyncio.create_task(coroutine)
        self._tasks.add(task)

        def finished(done: asyncio.Task) -> None:
            self._tasks.discard(done)
            self._cancelling.discard(done)
            if not done.cancelled():
                error = done.exception()
                if error is not None and self._is_current(turn_id) and self._failure is not None and not self._failure.done():
                    self._failure.set_result(error)

        task.add_done_callback(finished)
        return task

    def _is_current(self, turn_id: int) -> bool:
        return self._is_running and turn_id == self._turn_id

    def _request_cancel(self, *tasks: Optional[asyncio.Task]) -> None:
        for task in tasks:
            if task is not None and not task.done() and task not in self._cancelling:
                self._cancelling.add(task)
                task.cancel()

    async def run(self) -> None:
        """Run until disconnect, stop(), cancellation, or a provider failure."""
        if self._is_running:
            raise RuntimeError("This YapSesh is already running")
        self._is_running = True
        self._stop_event.clear()
        self._failure = asyncio.get_running_loop().create_future()
        try:
            async with self.stt.session(), self.tts.session():
                await self._run_connected()
        finally:
            self._is_running = False
            self._failure.cancel()
            self._tasks.clear()
            self._cancelling.clear()
            self._stt_queue = None
            self._is_speaking = False
            self._reslicer.clear()
            self._preroll.clear()
            self.vad.reset()

    async def _run_connected(self) -> None:
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
            self._request_cancel(*tasks)
            await asyncio.gather(*tasks, return_exceptions=True)

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
        await self.transport.send_event(SessionEvent(event_type, data))

    async def _process_frame(self, frame: bytes) -> None:
        event = self.vad.process(frame)
        if event == "start" and not self._is_speaking:
            self._is_speaking = True
            self._turn_id += 1
            # Retire the turn before any await. Provider cleanup runs in the
            # background; stale output is rejected even if cancellation is slow.
            self._request_cancel(self._active_reply_task, self._transcription_task)
            await self.transport.stop_playback()
            await self._emit("speech_started", turn_id=self._turn_id)
            queue: asyncio.Queue = asyncio.Queue()
            for past in self._preroll:
                queue.put_nowait(past)
            self._preroll.clear()
            queue.put_nowait(frame)
            self._stt_queue = queue
            self._transcription_task = self._spawn(self._run_transcription(queue, self._turn_id), self._turn_id)
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
        async def audio_stream() -> AsyncGenerator[bytes, None]:
            while True:
                item = await queue.get()
                if item is None:
                    break
                yield item

        committed = []
        interim = ""
        async with self._stt_lock, aclosing(audio_stream()) as audio:
            if not self._is_current(turn_id):
                return
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
                    if self._is_current(turn_id):
                        # A provider's final segment is not yet the full turn.
                        await self._emit("transcript", turn_id=turn_id, text=text, is_final=False)
            finally:
                close = getattr(stream, "aclose", None)
                if close is not None:
                    await close()
        text = " ".join([*committed, interim.strip()]).strip()
        if self._is_current(turn_id) and not self._is_speaking:
            await self._emit("transcript", turn_id=turn_id, text=text, is_final=True)
            if text and self._is_current(turn_id):
                self._active_reply_task = self._spawn(self._run_agent_reply(text, turn_id), turn_id)

    async def _run_agent_reply(self, user_prompt: str, turn_id: int) -> None:
        tokens: asyncio.Queue = asyncio.Queue(maxsize=128)
        response = ""
        completed = False
        status = "error"

        async def generate() -> None:
            nonlocal response
            async with self._agent_lock:
                if not self._is_current(turn_id):
                    return
                stream = self.agent.respond(user_prompt)
                try:
                    async for event in stream:
                        if not self._is_current(turn_id):
                            return
                        if event.type == "text_delta" and event.content:
                            response += event.content
                            await self._emit("agent_delta", turn_id=turn_id, text=event.content)
                            await tokens.put(event.content)
                    await tokens.put(None)
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()

        async def token_stream() -> AsyncGenerator[str, None]:
            while True:
                token = await tokens.get()
                if token is None:
                    break
                yield token

        async def speak() -> None:
            async with self._tts_lock, aclosing(token_stream()) as text:
                if not self._is_current(turn_id):
                    return
                stream = self.tts.synthesize_stream(text)
                try:
                    async for chunk in stream:
                        if not self._is_current(turn_id):
                            return
                        await self.transport.send_audio(chunk)
                finally:
                    close = getattr(stream, "aclose", None)
                    if close is not None:
                        await close()

        tasks = [asyncio.create_task(generate()), asyncio.create_task(speak())]
        try:
            # wait() leaves cancellation to our finally block, so a slow
            # provider's cleanup is not cancelled a second time by gather().
            pending = set(tasks)
            while pending:
                done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                for task in done:
                    task.result()
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
