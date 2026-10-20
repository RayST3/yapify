# Introducing YapWrap

**Make your agents yap.**

Give the agent you've already built a voice. YapWrap connects your existing
Python agent to speech recognition, speech synthesis, and live audio—so people
can talk to it.

You've chosen a framework, configured a model, connected tools, and worked out
memory and workflows. YapWrap starts with that agent. Wrap it in an adapter,
choose your speech providers, and turn it into a yapper.

Built-in adapters cover **Agno, Claude Agent SDK, CrewAI, Google ADK,
Hugging Face smolagents, LangGraph / LangChain agents, LlamaIndex,
Microsoft Agent Framework, OpenAI Agents SDK, and Pydantic AI**.
Custom Python agents can connect through `CallableAgent` or a `BaseAgent` subclass.
See [every adapter and its example](#bring-your-framework).

Start with any framework—or your own agent implementation. Here, `reply` is
your application's function that accepts user text and returns reply text or
a stream of text chunks:

```python
import asyncio

from yapwrap import YapWrap
from yapwrap.agents import CallableAgent
from yapwrap.stt import DeepgramSTT
from yapwrap.tts import CartesiaTTS
from yapwrap.vad import SileroVAD

from your_app import reply  # Your existing agent's text-in, text-out entry point.

yapper = YapWrap(
    agent=CallableAgent(reply),
    stt=DeepgramSTT(),
    tts=CartesiaTTS(),
    vad=SileroVAD(),
)

asyncio.run(yapper.startlocal())
```

Your function keeps calling your agent with its configured model, instructions,
tools, and session state. It can be synchronous or asynchronous. YapWrap handles
listening and speaking.

You can also wrap framework objects directly with a [built-in adapter](#bring-your-framework).
Install the voice dependencies below and keep your agent's own credentials and
session configuration.

**[Try it locally](#get-talking) · [Try it in your browser](#choose-where-your-agent-yaps) · [Find your framework](#bring-your-framework) · [Build an adapter](#make-it-your-own)**

## Why YapWrap?

Adding voice to an existing agent brings a new set of concerns: microphone
input, speech recognition, turn detection, audio playback, interruptions, and
keeping the conversation visible. Reusing an agent also means connecting its
input, output, and session state to that voice pipeline.

YapWrap puts that connection at the center of its API:

- **Bring your agent.** Use an adapter around your configured agent, graph,
  runner, or client. Your framework continues to execute its tools and workflows.
- **Choose your speech stack.** Configure STT, TTS, and voice activity detection
  independently of the agent.
- **Handle conversation turns.** Detect speech, turn reply text into spoken
  phrases, and clear queued playback when the user interrupts.
- **Talk locally or in a browser.** Use microphone and speakers, WebSockets,
  FastAPI, or WebRTC. The included clients show both sides of the conversation.
- **Extend through base classes.** Add an agent, provider, or transport by
  implementing a small, explicit interface.

The core imports no agent framework, web framework, audio device, or model SDK.
Integrations are optional.

```text
Your voice → speech recognition → your existing agent → speech synthesis → its voice
```

## Get talking

Python 3.10+ is required by YapWrap; individual framework SDKs can require a newer
version. From the repository root, install the example dependencies:

```sh
uv sync --group examples --extra local --extra web --extra silero --extra deepgram --extra cartesia
```

Set your provider keys. The demos use Agno with Gemini, Deepgram for recognition,
Cartesia for speech, and Silero for voice activity detection.

```sh
export GOOGLE_API_KEY="your-google-key"
export DEEPGRAM_API_KEY="your-deepgram-key"
export CARTESIA_API_KEY="your-cartesia-key"
```

<details>
<summary>PowerShell</summary>

```powershell
$env:GOOGLE_API_KEY = "your-google-key"
$env:DEEPGRAM_API_KEY = "your-deepgram-key"
$env:CARTESIA_API_KEY = "your-cartesia-key"
```

</details>

Start a local conversation:

```sh
uv run python -m examples.transports.local
```

Wait for `Session online`, then speak. The terminal prints your recognized
words and streams the agent's reply. Press Ctrl+C to stop.

`GEMINI_API_KEY` is also accepted. Set `GEMINI_MODEL` to change the example model;
the default is `gemini-flash-lite-latest`. These are example defaults—you choose
the model and providers in your own application.

For an application using this checkout, install only the integrations you need:

```sh
python -m pip install -e ".[local,silero,deepgram,cartesia]"
```

Install your agent's framework and model SDK separately. This command installs
YapWrap's voice dependencies; the `examples` group above also installs Agno and
Google's model SDK. See the [example guide](examples/README.md) for other setups.

## Choose where your agent yaps

| Experience | Example | Run from the repository root |
| --- | --- | --- |
| Microphone, speakers, and terminal transcript | [Local audio](examples/transports/local.py) | `uv run python -m examples.transports.local` |
| Browser voice chat and transcript | [FastAPI WebSocket](examples/transports/fastapi_websocket.py) | `uv run python -m examples.transports.fastapi_websocket` |
| A socket for your own client | [Standalone WebSocket](examples/transports/websocket.py) | `uv run python -m examples.transports.websocket` |
| Browser audio over WebRTC | [WebRTC](examples/transports/webrtc.py) | `uv run --extra webrtc python -m examples.transports.webrtc` |

For either browser server, open <http://localhost:8000>, allow microphone access,
and start a call. Run one at a time; both use port 8000. The standalone socket
listens on port 8765; its [browser client instructions](examples/README.md#transport-examples)
show how to serve the HTML separately.

Every example contains an explicit `yapper = YapWrap(...)`. Server examples
construct it inside the connection handler, giving each call its own agent,
speech components, and conversation state.

In your application, choose the matching entry point:

- `await yapper.startlocal()` for microphone and speakers.
- `await yapper.server_ws(ws=websocket)` for an accepted WebSocket connection.
- `await yapper.run(transport)` for a `BaseTransport` implementation.

A single `YapWrap` instance serves one conversation at a time.

## Bring your framework

Built-in adapters connect these Python agent APIs. Each linked example shows
the required SDK dependencies, object types, and session setup.

| Your agent | YapWrap adapter | Example |
| --- | --- | --- |
| Agno | `AgnoAgent` | [Agno](examples/frameworks/agno_agent.py) |
| LangGraph / LangChain agents | `LangGraphAgent` | [LangGraph](examples/frameworks/langgraph_agent.py) |
| Pydantic AI | `PydanticAIAgent` | [Pydantic AI](examples/frameworks/pydantic_ai_agent.py) |
| OpenAI Agents SDK | `OpenAIAgentsSDKAgent` | [OpenAI Agents](examples/frameworks/openai_agents_sdk.py) |
| Google ADK | `GoogleADKAgent` | [Google ADK](examples/frameworks/google_adk_agent.py) |
| Claude Agent SDK | `ClaudeAgentSDKAgent` | [Claude Agent SDK](examples/frameworks/claude_agent_sdk.py) |
| CrewAI | `CrewAIAgent` | [CrewAI](examples/frameworks/crewai_agent.py) |
| LlamaIndex | `LlamaIndexAgent` | [LlamaIndex](examples/frameworks/llamaindex_agent.py) |
| Microsoft Agent Framework | `MicrosoftAgentFrameworkAgent` | [Microsoft Agent Framework](examples/frameworks/microsoft_agent.py) |
| Hugging Face smolagents | `SmolAgentsAgent` | [smolagents](examples/frameworks/smolagents_agent.py) |
| Your own Python function or stream | `CallableAgent` | [Callable](examples/frameworks/callable_agent.py) |

Adapters translate recognized text into framework calls and reply text into
YapWrap events. They work with the interfaces shown in the examples; arbitrary
graph inputs, structured outputs, or application-specific events may need a
custom adapter. CrewAI and smolagents currently speak the final answer.

### Keep your framework's memory

Continue using your framework's session, history, and persistence mechanisms.
Pass the relevant run options through its adapter—for example, an Agno
`session_id`, a LangGraph `config` with a `thread_id`, or a LlamaIndex `ctx`.

YapWrap does not automatically migrate or populate agent memory. Its
`yapper.memory` records generated conversation turns and interruptions.
Your application decides which framework session each voice conversation uses.

## Where YapWrap fits

Pipecat and LiveKit can also connect existing agent logic.
[Pipecat includes a LangChain processor](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/processors/frameworks/langchain.py),
and [LiveKit provides a LangGraph/LangChain adapter](https://docs.livekit.io/agents/models/llm/langchain/).

YapWrap focuses on a direct Python integration: take an existing agent, wrap its
text interface, choose speech components, and run it through a transport.
It's built for developers who want to add voice while continuing to own their
agent's framework, tools, and session setup.

## Make it your own

Have an agent with a custom API? Use `CallableAgent` for a function that accepts
text and returns text or a stream of text. For more control, subclass `BaseAgent`:

```python
from collections.abc import AsyncIterator

from yapwrap import AgentEvent, BaseAgent

class MyAgent(BaseAgent):
    def __init__(self, existing_agent):
        self.existing_agent = existing_agent

    async def respond(self, text: str) -> AsyncIterator[AgentEvent]:
        # Adapt this call to your application's text-response API.
        reply = await self.existing_agent.reply(text)
        yield AgentEvent.text(reply)
```

Pass `MyAgent(existing_agent)` as the `agent` argument to `YapWrap`.
Stream `AgentEvent.text(delta)` as text arrives when your agent supports it.

The same approach applies to the rest of the pipeline:

| Base class | Implement |
| --- | --- |
| `BaseAgent` | `respond(text)` → async stream of agent events |
| `BaseSTT` | `transcribe(audio)` → async stream of transcripts |
| `BaseTTS` | `synthesize(text)` → async stream of PCM audio |
| `BaseVAD` | `process(pcm)` → `"start"`, `"stop"`, or `None` |
| `BaseTransport` | `audio_in()`, `send_audio(pcm)`, and `stop_playback()` |

TTS providers that accept incremental text can override
`synthesize_stream(text_deltas)` to produce audio directly from the agent's
token stream. The default implementation groups text into phrases and calls
`synthesize(text)`, so existing custom providers continue to work.

Built-in TTS adapters share three base classes, exported from `yapwrap`:

| Base class | What it handles | What an adapter implements |
| --- | --- | --- |
| `ManagedTTS` | One client per voice call, serialized access, cleanup, phrase fallback | `_open_connection()` and `synthesize(text)`; use `async with self._client()` for each request |
| `RealtimeTTS` | Managed lifetime plus incremental text and synthesis contexts across pauses | `_open_connection()` and `_synthesize_context(text_deltas)` |
| `WebSocketTTS` | Realtime streaming plus concurrent send/receive, timeouts, PCM alignment, and socket recovery | `_connection_settings()`, `_send_text(text_deltas, context_id)`, and `_decode_message(message, context_id)` |

`VoiceSession` opens `tts.session()` once for the voice call. Standalone synthesis
also cleans up its resources; wrap multiple standalone replies in
`async with tts.session():` to reuse a client. Create a separate TTS instance for
each concurrent call. A provider with a special SDK shutdown method can override
`_close_connection()`.

WebSocket decoders return `(pcm_bytes, finished)`. `finished` must mean that all
audio for the submitted context has arrived, including its final flush. Adapters
filter unrelated context IDs and raise provider errors. The default cancellation
closes the socket; `_cancel_context()` can retain it only when the protocol can
isolate cancelled audio from subsequent replies.

All five classes are exported from `yapwrap`. Start with the
[interface contracts](src/yapwrap/core/base.py) or run the
[custom components walkthrough](examples/custom_components.py) without keys or hardware:

```sh
uv run python -m examples.custom_components
```

Choose dependencies per integration: `local`, `web`, `websockets`, `webrtc`,
`silero`, `webrtc-vad`, `deepgram`, `cartesia`, `openai`, and `elevenlabs`
are available as extras; `audio` adds NumPy acceleration. The `all` extra
covers the listed runtime extras except `webrtc-vad`. Framework SDKs and
additional provider SDKs are installed separately.

### TTS streaming support

Sending text incrementally and receiving audio incrementally are separate
capabilities. The adapters use the following paths:

| Adapter | Text input and audio output | Resources between replies |
| --- | --- | --- |
| `CartesiaTTS` | Text deltas and PCM over WebSocket | Shared socket, separate context IDs |
| `ElevenLabsTTS` | Text deltas through multi-context WebSocket; flush then close each context | Shared socket, separate context IDs |
| `DeepgramAuraTTS` | `Speak` deltas, `Flush`, binary PCM, `Flushed` completion | Shared socket; reconnect after cancellation |
| `InworldTTS` | Bidirectional WebSocket, text deltas, raw PCM; wait for `contextClosed` | Shared socket; reconnect after cancellation |
| `RimeTTS` | Mist JSON WebSocket, text deltas, explicit flush, PCM | Shared socket; reconnect after cancellation |
| `SarvamTTS` | Bulbul WebSocket, text deltas, PCM, final completion event | Shared socket; reconnect after cancellation |
| `HumeTTS` | Octave TTS input stream; PCM snippets converted from 48 kHz | Reconnect each context after draining its normal server close |
| `AzureSpeechTTS` | SDK text stream with audio callbacks; PCM converted from 24 kHz | Shared synthesizer and SDK connection |
| `GoogleGeminiTTS` | Chirp3-HD: bidirectional text/audio; other voices: complete-phrase requests | Shared Cloud TTS client |
| `OpenAITTS` | Complete phrases into the Speech API; response bytes streamed and converted from 24 kHz | Shared HTTP client |

Rime, Inworld, Sarvam, and Hume need the `websockets` extra. Azure needs
`azure-cognitiveservices-speech>=1.37`. Google needs a current
`google-cloud-texttospeech` SDK with `StreamingAudioConfig` for streaming voices.
Despite its historical class name, `GoogleGeminiTTS` uses **Google Cloud TTS**.
Its default Journey voice keeps phrase synthesis; select a voice such as
`en-US-Chirp3-HD-Charon` for streaming input.

Most realtime adapters finish their current context after a 120 ms text pause;
Cartesia defaults to 750 ms. Adjust `context_idle_timeout` to suit token arrival
patterns. A tool pause flushes already generated text and later text starts a
new context. Rime also flushes at punctuation or after a delta takes the context
to 150 characters. Deepgram limits explicit flushes to 20 per 60 seconds, so it
flushes at pauses/end rather than every sentence and surfaces flush-limit errors.
This does not generate filler speech when the agent emits no text before a tool.

The streaming migration changes some provider defaults: Rime uses
`speaker="cove", model_id="mistv3"`; Inworld uses
`voice_id="Ashley", model="inworld-tts-2"`; Sarvam uses
`speaker="anushka", model="bulbul:v2"`; Hume uses the Octave voice `"Ava Song"`
instead of an EVI chat voice. Custom `base_url` values for these adapters must
target their streaming WebSocket endpoints. Supply compatible voices/models
explicitly if your application depends on a particular voice.

Protocol references:
[Cartesia](https://docs.cartesia.ai/api-reference/tts/websocket),
[ElevenLabs](https://elevenlabs.io/docs/eleven-api/guides/how-to/websockets/multi-context-web-socket),
[Deepgram flush semantics and limits](https://developers.deepgram.com/docs/tts-ws-flush),
[Inworld](https://docs.inworld.ai/api-reference/ttsAPI/texttospeech/synthesize-speech-websocket),
[Rime](https://docs.rime.ai/api-reference/mistv3/websockets-json),
[Sarvam](https://docs.sarvam.ai/api/api-guides-tutorials/text-to-speech/streaming-api/web-socket),
[Hume](https://dev.hume.ai/voice/reference/text-to-speech-tts/stream-input),
[Azure](https://learn.microsoft.com/azure/ai-services/speech-service/how-to-lower-speech-synthesis-latency),
[Google](https://docs.cloud.google.com/text-to-speech/docs/create-audio-text-streaming),
[OpenAI](https://developers.openai.com/api/docs/guides/text-to-speech).

## Working with real conversations

- **Audio:** components exchange raw PCM16 mono at a matching sample rate,
  defaulting to 16 kHz. Custom transports convert their audio at the boundary.
- **Transcripts:** local and browser clients display user and agent text.
  Custom transports can implement `send_event()` to receive `transcript`,
  `agent_delta`, and `agent_complete` events. The transcript records generated
  text; it does not confirm which audio the listener heard.
- **Interruptions:** a speech start cancels the current reply and clears queued
  playback. Microphone capture continues while providers clean up, and output
  from the retired turn is discarded. Each stateful component finishes cleanup
  before reuse. Synchronous work running in a Python thread cannot be forcibly
  stopped. Cancellation does not undo tool actions already performed.
- **Streaming speech:** adapters share connection and streaming engines, with
  provider-specific protocols and completion signals. Buffered text is flushed
  across tool pauses, and closed sockets reopen before a new context. See the
  [provider capabilities and migration notes](#tts-streaming-support).
- **Deployment:** the browser examples target localhost. Your application owns
  authentication, HTTPS, and any STUN/TURN configuration needed for WebRTC.
- **Validation:** automated checks cover the core, adapter behavior, provider
  protocol messages with mock connections, browser playback, and a local WebRTC
  exchange. Live provider compatibility and microphone/speaker quality still
  need testing with your configuration. Response time depends on the agent,
  speech providers, and network.

## Development

After installing the example dependencies:

```sh
uv run pytest -q -p no:cacheprovider
node tests/browser_client.test.cjs
uv build
```

Optional integration tests skip when their dependencies are absent. The build
produces a wheel and source archive that include the license.

## License

[Apache License, Version 2.0](LICENSE).
