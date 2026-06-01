---
status: proposed
contact: eavanvalkenburg
date: 2026-06-01
deciders: Agent Framework Python maintainers
consulted: Agent Framework Python maintainers
informed: Agent Framework contributors
---

# LLM-defined dynamic tools (self-extending agents)

## Context and Problem Statement

The Python function-calling loop now supports **progressive tool exposure**: a
tool, when invoked, receives a `FunctionInvocationContext` and can call
`context.add_tools(...)` / `context.remove_tools(...)` to change the set of
tools the model sees on the next loop iteration (see PR #6233, which supersedes
#3877 / #3398).

That mechanism only exposes tools that *already exist* in the host process. The
natural next step — requested by users building "skill library" / Voyager-style
agents — is to let the **model itself define a brand-new tool at runtime**: a
name, a description, a parameter schema, and an *implementation body*, which the
framework then compiles and exposes through the same `add_tools` mechanism.

This is powerful (an agent can grow its own capabilities mid-run) but the
implementation body is, by definition, **model-authored code**. The central
question is: *how do we let an LLM define and run new tools without turning the
agent into a remote-code-execution and data-exfiltration vector?*

## Decision Drivers

- **Security first.** Executing model-authored code is the highest-risk thing an
  agent can do. The default must be safe; unsafe paths must not exist in core.
- **Reuse existing isolation.** CodeAct providers (`agent-framework-monty`,
  `agent-framework-hyperlight`) already run model-authored code in a sandbox
  (Monty = Rust interpreter, FS/network blocked, `ResourceLimits`; Hyperlight =
  WASM micro-VM). We should build on that, not invent a new execution primitive.
- **Layering / dependency hygiene.** `agent-framework-core` must not depend on a
  sandbox runtime. The mechanism in core must be substrate-agnostic.
- **Human-in-the-loop.** A newly-defined capability should be approvable by a
  human before it runs.
- **Lifecycle correctness.** Dynamic tools must interact correctly with the
  approval pause/resume flow, checkpointing, and serialization.
- **Bounded behavior.** Self-extension must be bounded (no unbounded recursive
  tool generation, no tool-list explosion).
- **Incremental delivery.** Ship a small, secure, well-tested slice; defer the
  riskier and larger options behind clearly-scoped follow-ups.

## Considered Options

Two largely-independent axes: the **execution substrate** for the tool body, and
the **persistence lifetime** of a defined tool.

### Execution substrate

- **A. Sandbox-only** — the body runs in a CodeAct sandbox (Monty/Hyperlight).
- **B. Composition-only** — a "defined tool" is not arbitrary code at all; it is
  a declarative, parameterized composition of *existing, already-approved*
  tools.
- **C. In-process `exec()`** — compile the body into the host Python process
  (optionally behind a "restricted" namespace and an opt-in flag).

### Persistence lifetime

- **P1. Ephemeral** — the tool exists only for the current run.
- **P2. Session-scoped** — specs stored on session state, re-added each run.
- **P3. Durable skill library** — specs persisted across sessions for reuse.

## Decision Outcome

Chosen: **Substrate A (sandbox-only) + Persistence P1 (ephemeral)**, delivered as
a **layered design** with a substrate-agnostic core mechanism, behind a
**default-deny** policy flag, marked **experimental**.

- **Core (`agent_framework`)** ships the provider-agnostic mechanism only:
  `DynamicToolSpec`, a `ToolCodeCompiler` protocol returning a
  `CompiledDynamicTool` descriptor, a `DynamicToolPolicy` (default `enabled=
  False`), a run-local dynamic-tool registry, and a `make_define_tool(...)`
  factory that produces the `define_tool` meta-tool. Core ships **no executor
  that runs arbitrary code in-process**.
- **`agent-framework-monty`** ships `MontySandboxToolCompiler`, the recommended
  secure substrate, reusing the existing `InlineCodeBridge`.
- Defined tools default to `approval_mode="always_require"`; the policy also
  requires approval of the `define_tool` call itself.

Rationale: this resolves the security driver (no in-process RCE path exists in
core), respects layering (core stays sandbox-agnostic), and reuses the
already-reviewed CodeAct isolation. Ephemeral persistence is the only lifetime
that requires no new durable trust boundary, so it is the safe starting point.

**Explicitly deferred to follow-ups** (documented here, not implemented now):

- **Option B (composition)** — attractive because it needs no code execution,
  but in this framework `approval_mode` and middleware/accounting are enforced by
  the *function-calling loop*, not by `FunctionTool.invoke()`. A naive
  composition tool that calls sub-tools directly would **bypass approval,
  middleware, telemetry, and max-invocation accounting**. Doing it correctly
  requires routing sub-calls back through the loop's enforcement path; that is a
  separate, non-trivial design.
- **Option C (in-process exec)** — **rejected for core.** A "restricted" `exec`
  is not a security boundary in CPython; shipping it as a framework feature about
  model-authored code is an attractive RCE foot-gun. Users who genuinely want it
  can write their own `ToolCodeCompiler` outside the framework.
- **Hyperlight substrate** — mirrors the Monty compiler; follow-up.
- **P2 / P3 persistence** — session and durable skill libraries add a persistent
  trust boundary (poisoning / supply-chain risk) and need provenance, signing,
  and review gating; follow-up.

### Consequences

- Good: the only execution path that ships is sandboxed; core has no RCE surface.
- Good: core stays dependency-light and substrate-agnostic; other sandboxes
  plug in via the `ToolCodeCompiler` protocol.
- Good: reuses existing, reviewed CodeAct isolation and resource limits.
- Good: default-deny + experimental means zero behavior change unless explicitly
  opted in.
- Bad / accepted: the most "magical" variants (compose existing tools, persist a
  skill library) are not available yet.
- Bad / accepted: the feature is only useful to users who install a CodeAct
  provider; core alone cannot execute a defined tool body.

## Threat model and mitigations

| Threat | Vector | Mitigation |
|--------|--------|------------|
| Remote code execution | Model-authored body runs in host | No in-process executor in core; bodies run only in the sandbox substrate. |
| Prompt-injection-driven malicious tool | Injected instructions cause a harmful tool to be defined | Default-deny policy; human approval on **define** and **first invoke**; sandbox blocks FS/network/secrets; outbound access only via explicitly allow-listed host tools. |
| Confused deputy / privilege chaining | Defined tool calls powerful host tools | Compiled bodies get a **restricted tool_map** (no host tools unless explicitly granted); `FunctionInvocationContext` is **never** passed into a generated body; capabilities are granted explicitly, never ambient. |
| Approval bypass | Dynamic tool invoked without human sign-off | Default `approval_mode="always_require"`; approval payload surfaces name, schema, body/spec hash, provenance, compiler/substrate, and concrete args. |
| Recursive self-extension | A defined tool defines more tools, unbounded | Framework-computed definition depth + `max_definition_depth`; `max_dynamic_tools` cap; same-name/different-spec rejected. |
| Schema abuse (ReDoS, deep nesting) | Hostile JSON schema for params | Accept only a documented JSON-schema subset, with size and nesting-depth limits; reject unsupported features. |
| Secret exfiltration via telemetry | "Log the full spec" leaks secrets the model copied | Telemetry stores **hashes, sizes, compiler id, policy decision, provenance** by default; the full body is recorded only via an explicit secure audit sink behind a sensitive-data opt-in. |
| Broken approval resume / checkpoint | Dynamic tool/spec missing after a pause→resume round-trip | Persist the `DynamicToolSpec` + `compiler_id` + policy hash; **recompile and re-add** on resume. Never rely on closures or the prior mutable tool list surviving. |
| Name shadowing / spoofing | Redefining a trusted tool name (e.g. `send_email`) | Reserved-name list + no-override; `add_tools` already raises on a duplicate name with a different object. |
| Define races | Parallel `define_tool` calls in one model turn | Run-local registry with atomic check-and-add; identical name+spec is idempotent, name/different-spec is a deterministic policy error. |
| Provenance forgery | Model lies about depth/parent to dodge limits | Depth, parent call id, model/run id, and spec hash are computed in framework state, not taken from model input. |

## Validation

- Unit tests (folded into existing core/monty test files — no new test files):
  default-deny gating, approval required on define and invoke, reserved-name and
  no-override rejection, `max_dynamic_tools` and `max_definition_depth` caps,
  schema-subset rejection, the sandbox happy path (define → add → invoke), an
  approval pause→resume that recompiles from the persisted spec, and define-race
  idempotency.
- A sample demonstrating `define_tool` with the Monty sandbox compiler and the
  approval flow.
- Static checks: lint, mypy, pyright for `core` and `monty`.

## Pros and Cons of the Options

### A. Sandbox-only

- Good, because the execution boundary is a real sandbox with resource limits.
- Good, because it reuses already-reviewed CodeAct isolation.
- Neutral, because it requires installing a CodeAct provider.
- Bad, because core alone cannot run a defined tool body.

### B. Composition-only

- Good, because no new code is executed.
- Bad, because naive composition bypasses loop-enforced approval/middleware/
  accounting — i.e. it is *not* automatically safe.
- Bad, because doing it correctly (re-entering the enforcement path) is a
  separate, non-trivial design.

### C. In-process `exec()`

- Good, because it is trivial to implement and needs no extra dependency.
- Bad, because "restricted" `exec` is not a security boundary; it is effectively
  host RCE driven by model output.
- Bad, because shipping it in a framework normalizes an unsafe pattern.

### Persistence P1 / P2 / P3

- P1 (ephemeral): Good — no persistent trust boundary; safe default. Bad — tools
  vanish after the run.
- P2 (session): Good — reuse within a conversation. Bad — needs spec storage and
  re-validation each run.
- P3 (durable library): Good — true reusable skills. Bad — persistent poisoning /
  supply-chain risk; needs provenance, signing, and review gating.

## More Information

- Builds on the progressive tool exposure feature (PR #6233, superseding #3877
  and #3398) and `FunctionInvocationContext.add_tools` / `remove_tools`.
- Related: [0024-codeact-integration.md](0024-codeact-integration.md),
  [0024-prompt-injection-defense.md](0024-prompt-injection-defense.md),
  [0021-agent-skills-design.md](0021-agent-skills-design.md).
- The feature is shipped **experimental** and **default-deny**; revisit
  promotion, the composition substrate, Hyperlight support, and durable
  persistence once there is real-world usage feedback.
