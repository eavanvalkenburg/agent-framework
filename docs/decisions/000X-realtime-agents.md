---
# These are optional elements. Feel free to remove any of them.
status: proposed
contact: eavanvalkenburg
date: 2025-11-13
deciders: sergeymenshykh, markwallace, rbarreto, dmytrostruk, westey-m, eavanvalkenburg, stephentoub
consulted: taochenosu, alliscode, moonbox3
---

# Realtime/Voice Agents in the Agent Framework

## Context and Problem Statement

Currently we have support for Chat(Client)Agents in Agent Framework, and next to that a few agents that also follow a text-based approach. However, there is a need for agents that can handle streams of data both in and out as well, this is mostly for voice, but the current generation of voice api's also support text inputs and outputs. The key difference therefore is the presence of a input and/or output stream and these streams are not necessarily limited to voice. There are different ways for Agent Framework to support this, and this ADR captures the options considered.

## Decision Drivers

- Ease of use for developers wanting to build voice agents.
- Consistency with existing Agent Framework abstractions and patterns.
- Flexibility to support various input/output modalities (voice, text, etc.).
- Day 1 support for both OpenAI Realtime API, Azure Voice Live API and we should also be able to support [Google Gemini Live](https://gemini.google/overview/gemini-live/) to ensure we cover the major cloud providers.

## Considered Options

- RealtimeAgent abstraction (protocol or interface) with a concrete RealtimeAgent that takes a RealtimeClient (analogous to ChatAgent and ChatClient)
- RealtimeAgent abstraction (protocol or interface) with concrete RealtimeAgents directly implemented for providers.

## Pros and Cons of the Options

### RealtimeAgent with RealtimeClient

#### Pros
- Consistent with existing Agent Framework abstractions (ChatAgent/ChatClient).
- Clear separation of concerns between agent logic and communication protocol.
- Easier to extend support for additional providers in the future.
- Closer alignment of agent creation params (i.e. client, tools, instructions)
- Possibility to create a single ChatClient and RealtimeClient for a particular provider.

#### Cons
- Requires defining a new RealtimeClient abstraction, which may add complexity.
- Complexity in what happens within the RealtimeClient vs within the RealtimeAgent.

### RealtimeAgent with provider-specific implementations

#### Pros
- Simpler initial implementation without needing a new RealtimeClient abstraction.
- Direct control over provider-specific features and optimizations.
- Less indirection, potentially easier to understand for simple use cases.

#### Cons
- Divergence from existing Agent Framework patterns, leading to potential confusion.
- Harder to extend support for additional providers in the future.
- Mixing of agent logic with provider-specific communication details.
- More complex agent creation params (i.e. client, tools, instructions all in one place).

## Decision Outcome
