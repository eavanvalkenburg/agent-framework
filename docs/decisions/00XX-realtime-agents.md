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
**Comparison of Agent SDKs (Real-Time Voice Agent Support and Features)**

| **Agent SDK** | **Supports Realtime/Voice Agents?** | **Supported Providers** | **‘Sandwich’ STT→LLM→TTS Support** | **Programming Model (Inputs/Outputs)** | **Function Calling Support** | **Telemetry Support** | **Middleware/Callbacks Support** |
| ----------- | ---------- | ---------------- | --------- | -------- | -------- | ------- | -------- |
| **LangChain** | Yes (via external STT & TTS integration) [\[docs.langchain.com\]](https://docs.langchain.com/oss/python/langchain/voice-agent) | OpenAI, Azure, Cohere, Anthropic, Hugging Face, etc. | **Yes** – employs the “sandwich” pipeline (separate STT, text LLM agent, and TTS) [\[docs.langchain.com\]](https://docs.langchain.com/oss/python/langchain/voice-agent) | **Chain & agent API** – call chains/agents as functions (sync or async); final output returned via function call (token streaming supported via callbacks). | **Yes** (OpenAI function calling and tool use are supported) | **Partial** (logging and run tracing available, e.g. via LangSmith; no built-in metrics) | **Yes** (extensive callback handlers for events like token generation) |
| **OpenAI Agents SDK** | Yes (built for voice; uses OpenAI’s Realtime API for streaming audio) [\[platform.openai.com\]](https://platform.openai.com/docs/guides/voice-agents) | OpenAI only (GPT-4.1, GPT-4 Realtime models with voice) | **Yes** – supports both speech-to-speech and chained (Whisper STT → LLM → TTS) architectures [\[platform.openai.com\]](https://platform.openai.com/docs/guides/voice-agents), [\[platform.openai.com\]](https://platform.openai.com/docs/guides/voice-agents) | **Event-driven & async** – create a real-time session and handle streaming events (audio chunks, tool calls, etc.) in an async loop [\[openai.github.io\]](https://openai.github.io/openai-agents-python/realtime/quickstart/), [\[openai.github.io\]](https://openai.github.io/openai-agents-python/realtime/quickstart/). (TypeScript and Python SDKs available.) | **Yes** (built-in OpenAI function calling for tool use; agent outputs trigger tool execution) [\[openai.github.io\]](https://openai.github.io/openai-agents-python/realtime/quickstart/) | **No** (no built-in telemetry; developers can log events as needed) | **Yes** (exposes events for agent actions and audio, which developers handle for custom logic) [\[openai.github.io\]](https://openai.github.io/openai-agents-python/realtime/quickstart/) |
| **Google ADK (Agent Dev Kit)** | Yes (supports low-latency bidirectional voice with Gemini models) [\[cloud.google.com\]](https://cloud.google.com/blog/products/ai-machine-learning/build-a-real-time-voice-agent-with-gemini-adk) | Google Vertex AI (PaLM 2, Gemini, etc. – SDK in Python, TS, Go, Java) | **Yes** – built-in streaming speech pipeline (real-time STT input and TTS output via Vertex AI) [\[cloud.google.com\]](https://cloud.google.com/blog/products/ai-machine-learning/build-a-real-time-voice-agent-with-gemini-adk) | **Async multi-tasking** – concurrency for audio I/O (e.g. using Python asyncio) [\[cloud.google.com\]](https://cloud.google.com/blog/products/ai-machine-learning/build-a-real-time-voice-agent-with-gemini-adk). Agents defined via code (or config); multi-turn state can be maintained in-memory or via cloud (Interactions API). | **Yes** (tool use and function calling are supported – agents can invoke APIs/operations as tools) [\[cloud.google.com\]](https://cloud.google.com/blog/products/ai-machine-learning/build-a-real-time-voice-agent-with-gemini-adk) | **No** (no dedicated telemetry module; use Google Cloud’s logging/monitoring externally if needed) | **Partial** (no high-level callback API, but agent’s event loop can be managed in code to intercept steps) |
| **Semantic Kernel (MS)** | Yes (experimental real-time audio via OpenAI/Azure connectors) [\[Realtime A...soft Learn | Learn.Microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/realtime) | Azure OpenAI, OpenAI, Hugging Face, etc. (model-agnostic) [\[microsoft.github.io\]](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/) | **Yes** – through integrated speech services (OpenAI/Azure STT & TTS) [\[Realtime A...soft Learn | Learn.Microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/concepts/ai-services/realtime) | **Plugins/skills** model – invoke AI or native functions via the kernel (sync or async); can use planner for multi-step flows. Supports streaming responses (with real-time connectors). | **Yes** (native function/plugins auto-invoked via OpenAI function calls) [\[More advan...soft Learn | Learn.Microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/telemetry-advanced) | **Yes** (built-in OpenTelemetry-based logging, metrics, tracing) [\[Observabil...soft Learn | Learn.Microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/concepts/enterprise-readiness/observability/) | **Yes** (supports hooks/filters as middleware to monitor or alter agent behavior) [\[Introducti...soft Learn | Learn.Microsoft.com\]](https://learn.microsoft.com/en-us/semantic-kernel/overview/) |
| **Microsoft AutoGen** | No (text-based only – voice requires external STT/TTS pipeline) [\[github.com\]](https://github.com/microsoft/autogen/discussions/6553) | OpenAI (ChatGPT/GPT-4 by default; Azure OpenAI configurable) [\[microsoft.github.io\]](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/) | **No** (not natively – voice interface must be added externally if needed) [\[github.com\]](https://github.com/microsoft/autogen/discussions/6553) | **Multi-agent conversation** – define agents (e.g. AssistantAgent & UserProxy) and start a chat via function calls; the framework orchestrates the dialogue between agents and returns the result [\[microsoft.github.io\]](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/). (Primarily synchronous request-response loops, with agents able to act autonomously in between.) | **Yes** (supports tools as part of agent workflow, e.g. code execution, web search) [\[microsoft.github.io\]](https://microsoft.github.io/autogen/0.2/docs/Getting-Started/) | **No** (no built-in telemetry/analytics in this research-oriented framework) | **No** (no explicit callback hooks; customization is done by modifying/extending agent logic in code)|
| **CrewAI** | No (no native voice support) |  |  |  | |  |
| **Haystack (deepset)** | No (no native voice support) |  |  |  | |  |
| **HF Transformers Agents** | No (no native voice support) |  |  |  | |  |



## Decision Drivers

- Ease of use for developers wanting to build voice agents.
- Consistency with existing Agent Framework abstractions and patterns.
- Flexibility to support various input/output modalities (voice, text, etc.).
- Day 1 support for both OpenAI Realtime API, Azure Voice Live API and we should also be able to support [Google Gemini Live](https://gemini.google/overview/gemini-live/) to ensure we cover the major cloud providers.

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
- Allows for `sandwich` RealtimeAgents, reusing existing Speech-to-Text Clients, ChatClients, and Text-to-Speech Clients with the same abstraction.

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

## Decision Outcome
