# Introducing Yapify

**Make your agents yap.**

Give the agent you've already built a voice. Yapify connects your existing
Python agent to speech recognition, speech synthesis, and live audio—so people
can talk to it.

You've chosen a framework, configured a model, connected tools, and worked out
memory and workflows. Yapify starts with that agent. Wrap it in an adapter,
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

from yapify import YapWrap
from yapify.agents import CallableAgent
from yapify.stt import DeepgramSTT
from yapify.tts import CartesiaTTS
from yapify.vad import SileroVAD

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
tools, and session state. It can be synchronous or asynchronous. Yapify handles
listening and speaking.

You can also wrap framework objects directly with a [built-in adapter](#bring-your-framework).
Install the voice dependencies below and keep your agent's own credentials and
session configuration.

**[Try it locally](#get-talking) · [Try it in your browser](#choose-where-your-agent-yaps) · [Find your framework](#bring-your-framework) · [Build an adapter](#make-it-your-own)**

## Why Yapify?

Adding voice to an existing agent brings a new set of concerns: microphone
input, speech recognition, turn detection, audio playback, interruptions, and
keeping the conversation visible. Reusing an agent also means connecting its
input, output, and session state to that voice pipeline.

Yapify puts that connection at the center of its API:

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

Python 3.10+ is required by Yapify; individual framework SDKs can require a newer
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
Yapify's voice dependencies; the `examples` group above also installs Agno and
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

| Your agent | Yapify adapter | Example |
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
Yapify events. They work with the interfaces shown in the examples; arbitrary
graph inputs, structured outputs, or application-specific events may need a
custom adapter. CrewAI and smolagents currently speak the final answer.

### Keep your framework's memory

Continue using your framework's session, history, and persistence mechanisms.
Pass the relevant run options through its adapter—for example, an Agno
`session_id`, a LangGraph `config` with a `thread_id`, or a LlamaIndex `ctx`.

Yapify does not automatically migrate or populate agent memory. Its
`yapper.memory` records generated conversation turns and interruptions.
Your application decides which framework session each voice conversation uses.

## Where Yapify fits

Pipecat and LiveKit can also connect existing agent logic.
[Pipecat includes a LangChain processor](https://github.com/pipecat-ai/pipecat/blob/main/src/pipecat/processors/frameworks/langchain.py),
and [LiveKit provides a LangGraph/LangChain adapter](https://docs.livekit.io/agents/models/llm/langchain/).

Yapify focuses on a direct Python integration: take an existing agent, wrap its
text interface, choose speech components, and run it through a transport.
It's built for developers who want to add voice while continuing to own their
agent's framework, tools, and session setup.

## Make it your own

Have an agent with a custom API? Use `CallableAgent` for a function that accepts
text and returns text or a stream of text. For more control, subclass `BaseAgent`:

```python
from collections.abc import AsyncIterator

from yapify import AgentEvent, BaseAgent

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

All five classes are exported from `yapify`. Start with the
[interface contracts](src/yapify/core/base.py) or run the
[custom components walkthrough](examples/custom_components.py) without keys or hardware:

```sh
uv run python -m examples.custom_components
```

Choose dependencies per integration: `local`, `web`, `websockets`, `webrtc`,
`silero`, `webrtc-vad`, `deepgram`, `cartesia`, `openai`, and `elevenlabs`
are available as extras; `audio` adds NumPy acceleration. The `all` extra
covers the listed runtime extras except `webrtc-vad`. Framework SDKs and
additional provider SDKs are installed separately.

## Working with real conversations

- **Audio:** components exchange raw PCM16 mono at a matching sample rate,
  defaulting to 16 kHz. Custom transports convert their audio at the boundary.
- **Transcripts:** local and browser clients display user and agent text.
  Custom transports can implement `send_event()` to receive `transcript`,
  `agent_delta`, and `agent_complete` events. The transcript records generated
  text; it does not confirm which audio the listener heard.
- **Interruptions:** a speech start cancels the current reply and clears queued
  playback. Synchronous work running in a Python thread cannot be forcibly
  stopped. Cancellation does not undo tool actions already performed.
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
