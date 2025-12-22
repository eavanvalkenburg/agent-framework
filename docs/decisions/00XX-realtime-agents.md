---
status: proposed
contact: eavanvalkenburg
date: 2025-12-22
deciders: sergeymenshykh, markwallace, rbarreto, dmytrostruk, westey-m, eavanvalkenburg, stephentoub
consulted: taochenosu, alliscode, moonbox3
---

# Realtime/Voice Agents in the Agent Framework

## Context and Problem Statement

Currently we have support for Chat(Client)Agents in Agent Framework, and next to that a few agents that also follow a text-based approach. However, there is a need for agents that can handle streams of data both in and out as well, this is mostly for voice, but the current generation of voice api's also support text inputs and outputs. The key difference therefore is the presence of a input and/or output stream and these streams are not necessarily limited to voice. There are different ways for Agent Framework to support this, and this ADR captures the options considered.

## Competitive landscape

> [!Note]
> “Realtime” here means low‑latency bidirectional audio. “Sandwich/chained” means separate STT and TTS components around a text‑centric agent. “Speech‑to‑speech” means a single multimodal model handles audio in/out natively.

***

### Comparison table

| SDK | Realtime/voice agent support | Supported realtime **providers** | “Sandwich/chained” STT→LLM→TTS | Programming model for inputs/outputs (realtime) | Function calling / tools | Telemetry / tracing | Middleware / callbacks |
| - | - | - | - | - | - | - | - |
| **LangChain** | **Yes** — official **Voice Agent** guide and demos. [\[docs.langchain.com\]](https://docs.langchain.com/oss/python/langchain/voice-agent) | Demonstrated with **AssemblyAI (STT)** + **ElevenLabs**/**Cartesia (TTS)**; LLM via OpenAI/Anthropic. Pluggable to other providers. [\[docs.langchain.com\]](https://docs.langchain.com/oss/python/langchain/voice-agent), [\[github.com\]](https://github.com/langchain-ai/voice-sandwich-demo), [\[deepwiki.com\]](https://deepwiki.com/langchain-ai/voice-sandwich-demo/3.1-voice-agent-pipeline) | **Yes** — documented “**Sandwich**” architecture; also discusses native speech‑to‑speech trade‑offs. [\[docs.langchain.com\]](https://docs.langchain.com/oss/python/langchain/voice-agent) | Browser ↔ server **WebSocket/WebRTC**; server composes **async generator** pipeline that streams **STT chunks → agent tokens → tool calls → TTS audio**. [\[deepwiki.com\]](https://deepwiki.com/langchain-ai/voice-sandwich-demo/3.1-voice-agent-pipeline) | **Yes** — agents can call tools; voice demo emits **tool\_call/tool\_result** events. [\[deepwiki.com\]](https://deepwiki.com/langchain-ai/voice-sandwich-demo/3.1-voice-agent-pipeline) | **Yes** — **LangSmith** tracing incl. **OpenTelemetry** export. [\[docs.langchain.com\]](https://docs.langchain.com/langsmith/trace-with-opentelemetry), [\[docs.langchain.com\]](https://docs.langchain.com/langsmith/trace-with-langchain) | **Yes** — built‑in **CallbackManager**, custom handlers, run grouping. [\[langchain....thedocs.io\]](https://langchain.readthedocs.io/en/latest/core/callbacks/langchain_core.callbacks.manager.trace_as_chain_group.html), [\[github.com\]](https://github.com/langchain-ai/langchain/blob/master/libs/core/langchain_core/callbacks/manager.py) |
| **Semantic Kernel (SK)** | **Yes** — SK Python adds **Realtime clients** for **OpenAI** & **Azure OpenAI** (WebRTC/WebSocket). [\[devblogs.m...rosoft.com\]](https://devblogs.microsoft.com/semantic-kernel/talk-to-your-agents-introducing-the-realtime-apis-in-semantic-kernel/), [\[github.com\]](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/realtime/README.md) | **OpenAI Realtime API** (e.g., *gpt‑realtime*), **Azure OpenAI GPT Realtime** (Foundry). [\[github.com\]](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/realtime/realtime_agent_with_function_calling_webrtc.py), [\[learn.microsoft.com\]](https://learn.microsoft.com/en-us/azure/ai-foundry/openai/how-to/realtime-audio?view=foundry-classic) | **Possible** — SK doesn’t ship STT/TTS; you can chain SK kernels with external STT/TTS for a sandwich pipeline, or use speech‑to‑speech via provider. Samples focus on S2S + function calling. [\[github.com\]](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/realtime/README.md) | **Client classes** (e.g., `OpenAIRealtimeWebRTC`, `AzureRealtimeWebRTC`) accept **PCM16 audio** input and stream **audio + transcripts + text events** back; event types include **RealtimeTextEvent**. [\[github.com\]](https://github.com/microsoft/semantic-kernel/blob/main/python/samples/concepts/realtime/realtime_agent_with_function_calling_webrtc.py) | **Yes** — use **FunctionChoiceBehavior** (auto) for tool/function calling; SK can **auto‑invoke** and serialize functions to the Realtime API. [\[devblogs.m...rosoft.com\]](https://devblogs.microsoft.com/semantic-kernel/talk-to-your-agents-introducing-the-realtime-apis-in-semantic-kernel/) | **Yes** — **OpenTelemetry** logs/metrics/traces across kernels/plugins/connectors. [\[learn.microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/), [\[github.com\]](https://github.com/microsoft/semantic-kernel/blob/main/dotnet/docs/TELEMETRY.md) | **Yes** — **Filters & hooks** (Function Invocation, Prompt Render, Auto Function Invocation) act as middleware; pre/post hooks available. [\[learn.microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/filters), [\[kerneljs.com\]](https://kerneljs.com/concepts/filters), [\[github.com\]](https://github.com/microsoft/semantic-kernel/blob/main/docs/decisions/0005-kernel-hooks-phase1.md) |
| **OpenAI Agents SDK** | **Yes** — first‑class **Realtime Agents**; GA **Realtime API** supports **WebRTC/WebSocket/SIP**. [\[platform.openai.com\]](https://platform.openai.com/docs/guides/realtime), [\[openai.github.io\]](https://openai.github.io/openai-agents-js/guides/voice-agents/transport/), [\[platform.openai.com\]](https://platform.openai.com/docs/api-reference/realtime-calls?_clear=true) | **OpenAI Realtime models** (e.g., *gpt‑realtime*), client **WebRTC** in browser, **WebSocket** server, **SIP** for telephony. [\[openai.github.io\]](https://openai.github.io/openai-agents-js/guides/voice-agents/transport/) | **Speech‑to‑speech** recommended; “**Chained**” architecture documented as an alternative pattern. [\[platform.openai.com\]](https://platform.openai.com/docs/guides/voice-agents) | **`RealtimeAgent` + `RealtimeSession`**: connect, stream mic audio (**PCM16**), receive **audio** and **transcripts**; **ephemeral keys** or unified calls for session creation. Event‑driven (**`session.on('audio')`**). [\[openai.github.io\]](https://openai.github.io/openai-agents-js/guides/voice-agents/build/), [\[platform.openai.com\]](https://platform.openai.com/docs/guides/realtime-webrtc) | **Yes** — **tools** and **handoffs** inside Realtime Agents. [\[openai.github.io\]](https://openai.github.io/openai-agents-js/guides/voice-agents/build/) | **Yes** — server‑side **tracing** on Realtime **Calls**; Agents SDK docs emphasize built‑in tracing/monitoring. [\[platform.openai.com\]](https://platform.openai.com/docs/api-reference/realtime-calls?_clear=true), [\[aurelio.ai\]](https://www.aurelio.ai/learn/openai-agents-sdk) | **Yes** — event listeners & transport layers act like middleware; pluggable **WebRTC/WebSocket/SIP** transports. [\[openai.github.io\]](https://openai.github.io/openai-agents-js/guides/voice-agents/transport/) |
| **Google Agents SDK (ADK)** | **Yes** — **Gemini Live API** for native speech‑to‑speech; ADK **streaming agents** and **Vertex AI Agent Engine** for managed hosting. [\[ai.google.dev\]](https://ai.google.dev/gemini-api/docs/live), [\[google.github.io\]](https://google.github.io/adk-docs/get-started/streaming/), [\[docs.cloud...google.com\]](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/live-api) | **Gemini Live API** (Google AI Studio/Vertex); ADK integrates with **LiveKit**/**Pipecat**/**Voximplant** and can run on **Vertex AI Agent Engine**. [\[ai.google.dev\]](https://ai.google.dev/gemini-api/docs/live), [\[discuss.google.dev\]](https://discuss.google.dev/t/deploy-bidirectional-streaming-agents-with-vertex-ai-agent-engine-and-live-api/266087) | **Primarily speech‑to‑speech**; ADK content contrasts classic **STT→LLM→TTS** pipelines with Live API’s native audio (you can still compose a chained approach if needed). [\[discuss.google.dev\]](https://discuss.google.dev/t/beyond-the-chatbot-webrtc-gemini-and-your-first-real-time-voice-agent/286679) | **Stateful WebSocket** streams: input **PCM16 16 kHz** audio, output **PCM16 24 kHz** audio; client‑to‑server or server‑to‑server, **ephemeral tokens**; ADK exposes **`Runner.runLive()`**/**`LiveRequestQueue`**/**`Flowable<Event>`**.  | **Yes** — Live API supports **tool use/function calling**; ADK ships built‑in tools (e.g., **GoogleSearch**, **MCP Toolset**).  | **Yes (platform)** — run on Vertex AI with enterprise monitoring/logging; Live API/Agent Engine integrate with Google Cloud ops. citeturn1search39turn1search38 | **Yes** — ADK is **event‑driven** with observable streams and callback‑style handlers. |


## Decision Drivers

- Ease of use for developers wanting to build voice agents.
- Consistency with existing Agent Framework abstractions and patterns.
- Flexibility to support various input/output modalities (voice, text, etc.).
- Day 1 support for both OpenAI Realtime API, Azure Voice Live API and we should also be able to support [Google Gemini Live](https://gemini.google/overview/gemini-live/) to ensure we cover the major cloud providers.
- Should easily support both Websockets and WebRTC based implementations.
- Should easily support `sandwich`/`chained` agents that combine Speech-to-Text, Chat, and Text-to-Speech components.

## Considered Options

- RealtimeAgent abstraction (protocol or interface) with a concrete RealtimeAgent that takes a RealtimeClient (analogous to ChatAgent and ChatClient)
- RealtimeAgent abstraction (protocol or interface) with concrete RealtimeAgents directly implemented per provider.

## Pros and Cons of the Options

### RealtimeAgent with RealtimeClient

#### Pros
- Consistent with existing Agent Framework abstractions (ChatAgent/ChatClient).
- Clear separation of concerns between agent logic and communication protocol.
- Easier to extend support for additional providers in the future as the implementation details are encapsulated in the RealtimeClient.
- Closer alignment of agent creation params (i.e. client, tools, instructions)
- Possibility to create a single ChatClient and RealtimeClient for a particular provider.
- Allows for `sandwich`/`chained` agents, reusing existing Speech-to-Text Clients, ChatClients, and Text-to-Speech Clients with the same (potentially base-)abstraction.

#### Cons
- Requires defining a new RealtimeClient abstraction, which may add complexity.
- Complexity in what happens within the RealtimeClient vs within the RealtimeAgent.
- Potentially slower as more layers are added.

### RealtimeAgent with provider-specific implementations

#### Pros
- Simpler initial implementation without needing two new RealtimeClient abstractions.
- Direct control over provider-specific features and optimizations.
- Less indirection, potentially easier to understand for simple use cases.

#### Cons
- Divergence from existing Agent Framework patterns, leading to potential confusion.
- Harder to extend support for additional providers in the future.
- Mixing of agent logic with provider-specific communication details.
- More complex agent creation params (i.e. client, tools, instructions all in one place).
- Would require additional specific implementations for `sandwich`/`chained` agents, leading to code duplication.

## Decision Outcome
