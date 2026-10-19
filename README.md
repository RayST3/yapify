# Yapify

A Python voice pipeline with pluggable agent, speech providers, and transport.
Python 3.10 or later is required.

```text
PCM input -> VAD -> streaming STT -> agent events -> phrase chunking -> TTS -> PCM output
```

The core imports no Agno, FastAPI, device, or model SDK. An agent only needs
`respond(text)`, an async iterator of `AgentEvent` objects. `CallableAgent` adapts
ordinary Python functions and streams; `AgnoAgent` adapts an existing Agno agent.

## Run the browser example

From this checkout in PowerShell:

```powershell
uv sync --group examples --extra web --extra silero --extra deepgram --extra cartesia
$env:DEEPGRAM_API_KEY = "your-deepgram-key"
$env:CARTESIA_API_KEY = "your-cartesia-key"
$env:GOOGLE_API_KEY = "your-google-key"
uv run python -m examples.transports.fastapi_websocket
```

Open <http://localhost:8000>, allow microphone access, and start a call. The
server also supports `uv run uvicorn examples.transports.fastapi_websocket:app --reload`.
The example uses an Agno agent with Gemini, Deepgram Nova transcription, and
Cartesia speech synthesis. All three provider keys are required; `GEMINI_API_KEY`
is also accepted in place of `GOOGLE_API_KEY`. `GEMINI_MODEL` overrides the model
ID (default `gemini-flash-lite-latest`). There is no echo fallback in this example.

The `examples` dependency group installs Agno and Google's model SDK for this
checkout. Every transport example uses `AgnoAgent` and Gemini Flash Lite by
default. [Framework examples](examples/frameworks) cover all supported adapters,
including a plain callable. [Transport examples](examples/transports) cover
local audio, standalone WebSockets, FastAPI WebSockets, and WebRTC. See the
[example guide](examples/README.md) for dependencies, credentials, and commands.

`web` installs FastAPI, Uvicorn, and websockets. `silero` installs the actual
Silero model package and its runtime dependencies. Installing only the base
package does not install these integrations. Silero buffers 20 ms pipeline
frames into the 512-sample windows required by its 16 kHz model.

The client captures mono PCM16 with an AudioWorklet, schedules output chunks in
order, and clears queued playback on interruption. Each connection receives a
new wrapper, VAD, providers, and history. Missing keys and session failures are
reported to the client, with error details in the server log.

The conversation panel shows your recognized words as they arrive and streams
the agent's reply into a separate message. Final transcript updates replace the
interim text, and interrupted replies are labeled. The transcript stays visible
after a call ends and clears when you start a new call.

## Install as a separate package

The project builds a regular wheel; it does not depend on an Agno checkout.

```powershell
python -m pip install -e ".[web,silero,deepgram,cartesia]"
# To run the Agno examples when installing with pip:
python -m pip install "agno>=3.1.5" "google-genai>=2.31.0"
# Or build a distributable artifact:
uv build
# Then, in another environment:
python -m pip install "dist/yapify-0.1.0-py3-none-any.whl[web,silero,deepgram,cartesia]"
```

Select only the extras needed by your application:

| Extra | Purpose |
| --- | --- |
| `web` | FastAPI browser example |
| `websockets` | Standalone WebSocket server |
| `local` | Microphone and speakers via sounddevice |
| `silero` | Neural voice activity detection |
| `deepgram`, `cartesia` | Providers used in the examples |
| `audio` | Optional NumPy acceleration |
| `webrtc`, `webrtc-vad`, `openai`, `elevenlabs` | Additional integration dependencies |

Agent SDKs are installed separately (or with `uv sync --group examples` for
the Agno/Gemini examples). `all` includes the listed runtime extras
except `webrtc-vad`; it does not install every third-party agent/provider SDK.

## Use any agent and transport

```python
from yapify import YapWrap
from yapify.agents import CallableAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import EnergyVAD

async def reply(text: str):
    yield f"You said: {text}"

yapper = YapWrap(
    agent=CallableAgent(reply),
    stt=DeepgramSTT(),
    tts=CartesiaTTS(),
    vad=EnergyVAD(),
)

# In an async entry point, choose one:
await yapper.startlocal()                 # requires the local extra
await yapper.server_ws(ws=accepted_ws)    # socket accepted by your application
await yapper.run(your_transport)          # no web framework required
```

Local sessions print the conversation automatically, including the Agno and
other framework examples:

```text
You: What can you help me with?
Agent: I can answer questions and help you plan. [interrupted]
You: Tell me about the weather.
Agent: Which city are you in?
```

User lines print once recognition finishes; agent text streams as it is
generated. Use `await yapper.startlocal(show_transcripts=False)` for audio only.

A custom transport implements:

```python
class MyTransport:
    async def audio_in(self):
        # Yield raw PCM16 mono bytes until the connection ends.
        yield ...

    async def send_audio(self, pcm: bytes) -> None:
        ...

    async def stop_playback(self) -> None:
        ...
```

All components use signed little-endian PCM16 mono at the same sample rate
(default 16 kHz). Incoming chunks may have arbitrary sizes; the session frames
them into 20 ms blocks. Components declaring a different `sample_rate` are
rejected rather than silently changing audio speed. Raw PCM has no WAV header.

For a standalone WebSocket server, use `await yapper.server_ws()`. To handle
concurrent connections, pass `session_factory=create_yapper`, where the factory
constructs independent components. A single wrapper cannot serve concurrent
calls. Applications own authentication, socket acceptance, and connection setup.

Transports can optionally implement `async send_event(event: SessionEvent)`
(`SessionEventSink`) to receive conversation text. Existing audio-only
transports need no changes. WebSocket clients receive JSON alongside PCM:

```json
{"type": "transcript", "data": {"turn_id": 1, "text": "Hello", "is_final": true}, "timestamp": 0.0}
{"type": "agent_delta", "data": {"turn_id": 1, "text": "Hi there!"}, "timestamp": 0.0}
{"type": "agent_complete", "data": {"turn_id": 1, "text": "Hi there!", "status": "completed"}, "timestamp": 0.0}
```

Transcript messages contain the whole recognized turn so far; agent deltas are
appended. `agent_complete` contains the whole generated reply and a status of
`completed`, `interrupted`, or `error`. Speech boundary events also include the
turn ID. These are recognized/generated words, not a record of audio actually
heard by the listener.

## Wrap Agno

Install Agno and the SDK for your chosen model separately, then pass your
existing agent:

```python
from yapify.agents import AgnoAgent

voice_agent = AgnoAgent(my_agent, session_id="conversation-123")
yapper = YapWrap(agent=voice_agent, stt=stt, tts=tts, vad=vad)
await yapper.run(transport)
```

See `examples/frameworks/agno_agent.py` for a local Agno/Gemini example. The core never
calls Agno directly; `AgnoAgent` translates streamed run events at the boundary.
Async agents support cancellation; a synchronous callable running in a worker
thread cannot be forcibly stopped.

## Lifecycle and transcript contract

A speech start cancels the previous reply/transcription and clears playback.
Each utterance owns its input queue, including preroll exactly once. Input EOF,
`VoiceSession.stop()`, or external cancellation cleans up pending tasks. STT,
agent, and TTS failures propagate to the application instead of silently ending
one half of the call.

Custom STT adapters yield `TranscriptEvent(text, is_final=True)` for committed
segments and `is_final=False` for interim replacements. Plain strings remain
supported as cumulative utterance snapshots. Adapters must finish after input
EOF and flush their final transcript; Deepgram uses `CloseStream` for this.

`yapper.memory` records generated turns and interruptions. It is an audit log,
not automatic context injection into every agent framework. Configure the
agent's own history/session storage, or explicitly use `memory.get_messages()`
in your callable. Sending audio does not prove it was heard: this transport
protocol has no playback acknowledgements, and the session does not populate
`user_heard` with guesses.

## Coverage and limitations

Regression coverage exercises the core pipeline and framework adapters,
Silero framing, Deepgram/Cartesia protocol messages with fake sockets, and
WebSocket/local playback behavior, and a local aiortc audio/transcript exchange.
Real provider calls and microphone/speaker
quality require a live smoke test with credentials and hardware.

The WebRTC example includes browser signaling and conversation text. Its host
ICE configuration is intended for localhost; deployments configure their own
STUN/TURN and authentication. Framework examples use current SDK interfaces,
but have not all been exercised with live providers. There is no universal
latency guarantee; model, provider, and network choices affect response time.

## Development

```powershell
uv sync --group examples --extra web --extra silero --extra deepgram --extra cartesia
uv run pytest -q -p no:cacheprovider
node tests/browser_client.test.cjs
```

Optional integration tests skip when their dependencies are absent. The core
also has an import test with site-packages disabled to catch framework coupling.
