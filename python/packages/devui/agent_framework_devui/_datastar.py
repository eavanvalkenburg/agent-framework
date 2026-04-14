# Copyright (c) Microsoft. All rights reserved.

"""Datastar streaming adapter for DevUI.

Transforms Agent Framework executor events into Datastar SSE events
(patch-elements, patch-signals) for server-driven UI updates. Renders
markdown to HTML on the server so the browser receives ready-to-display
DOM fragments instead of JSON that needs client-side processing.
"""

from __future__ import annotations

import html
import json
import logging
import uuid
from collections.abc import AsyncGenerator
from datetime import datetime, timezone
from typing import Any

import mistune

from ._executor import AgentFrameworkExecutor
from .models import AgentFrameworkRequest

logger = logging.getLogger(__name__)


# region Markdown renderer

_markdown_renderer: mistune.Markdown | None = None


def _get_markdown_renderer() -> mistune.Markdown:
    """Return a cached mistune Markdown renderer."""
    global _markdown_renderer
    if _markdown_renderer is None:
        _markdown_renderer = mistune.create_markdown(
            plugins=["table", "strikethrough", "task_lists", "spoiler"],
        )
    return _markdown_renderer


def render_markdown(text: str) -> str:
    """Render markdown text to HTML."""
    md = _get_markdown_renderer()
    result = md(text)
    if result is None:
        return ""
    return str(result)


# region Avatar SVG helpers

_BOT_AVATAR_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
    '<rect width="18" height="10" x="3" y="11" rx="2"/>'
    '<circle cx="12" cy="5" r="2"/><path d="M12 7v4"/>'
    '<line x1="8" x2="8" y1="16" y2="16"/>'
    '<line x1="16" x2="16" y1="16" y2="16"/></svg>'
)

_USER_AVATAR_SVG = (
    '<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"'
    ' stroke-linecap="round" stroke-linejoin="round" width="16" height="16">'
    '<circle cx="12" cy="8" r="5"/>'
    '<path d="M20 21a8 8 0 0 0-16 0"/></svg>'
)


_BOT_AVATAR_HTML = f'<div class="avatar bot-avatar">{_BOT_AVATAR_SVG}</div>'
_USER_AVATAR_HTML = f'<div class="avatar user-avatar">{_USER_AVATAR_SVG}</div>'


def _bot_avatar() -> str:
    """Return the bot avatar HTML element with SVG icon."""
    return _BOT_AVATAR_HTML


def _user_avatar() -> str:
    """Return the user avatar HTML element with SVG icon."""
    return _USER_AVATAR_HTML


# region Datastar SSE helpers


def _patch_elements(element_html: str) -> str:
    """Format a ``datastar-patch-elements`` SSE event (morph mode)."""
    lines = element_html.split("\n")
    data_lines = "\n".join(f"data: elements {line}" for line in lines)
    return f"event: datastar-patch-elements\n{data_lines}\n\n"


def _append_elements(selector: str, element_html: str) -> str:
    """Format a ``datastar-patch-elements`` SSE event in append mode."""
    lines = element_html.split("\n")
    data_lines = "\n".join(f"data: elements {line}" for line in lines)
    return f"event: datastar-patch-elements\ndata: selector {selector}\ndata: mode append\n{data_lines}\n\n"


def _prepend_elements(selector: str, element_html: str) -> str:
    """Format a ``datastar-patch-elements`` SSE event in prepend mode."""
    lines = element_html.split("\n")
    data_lines = "\n".join(f"data: elements {line}" for line in lines)
    return f"event: datastar-patch-elements\ndata: selector {selector}\ndata: mode prepend\n{data_lines}\n\n"


def _patch_signals(signals: dict[str, Any]) -> str:
    """Format a ``datastar-patch-signals`` SSE event."""
    return f"event: datastar-patch-signals\ndata: signals {json.dumps(signals)}\n\n"


# region Time helper


def _now_str() -> str:
    """Return current local time as h:mm:ss AM/PM."""
    return datetime.now(tz=timezone.utc).astimezone().strftime("%I:%M:%S %p").lstrip("0")


def _token_display(total: int, inp: int, out: int) -> str:
    """Unified token display: ``409 ↑345 ↓64``."""
    return f'{total:,} <span class="tok tok-in">&uarr;{inp:,}</span> <span class="tok tok-out">&darr;{out:,}</span>'


# region Sidebar event & tool formatting

_EVT_COLORS: dict[str, str] = {
    "response.output_text.delta": "gray",
    "response.output_text.done": "gray",
    "response.output_item.added": "green",
    "response.output_item.done": "green",
    "response.content_part.added": "green",
    "response.function_call_arguments.delta": "blue",
    "response.function_call.complete": "blue",
    "response.function_result.complete": "green",
    "response.completed": "green",
    "response.failed": "red",
    "response.error": "red",
    "response.reasoning_text.delta": "orange",
    "response.trace.completed": "orange",
    "response.workflow_event.completed": "purple",
    "response.cancelled": "red",
}


def _evt_entry(event_type: str, summary: str, seq: int, *, call_id: str = "") -> str:
    """One sidebar event row matching the React UI style.

    When *call_id* is provided (for function events), the row becomes
    clickable and navigates to the corresponding tool card.
    """
    color = _EVT_COLORS.get(event_type, "gray")
    badge = event_type.replace("response.", "").replace(".", "_")
    eid = f"evt-{seq}"
    ts = _now_str()
    summary = (summary[:90] + "…") if len(summary) > 90 else summary

    # Clickable link to tool entry
    click_attr = ""
    link_class = ""
    if call_id:
        click_attr = (
            f" data-on:click=\"$activeTab = 'tools'; "
            f"setTimeout(() => document.getElementById('tool-entry-{call_id}')"
            f"?.scrollIntoView({{behavior:'smooth',block:'center'}}), 100)\""
        )
        link_class = " evt-link"

    return (
        f'<div id="{eid}" class="evt-row{link_class}"{click_attr}>'
        f'<span class="evt-dot dot-{color}"></span>'
        f'<span class="evt-ts">{ts}</span>'
        f'<span class="evt-badge">{html.escape(badge)}</span>'
        f'<div class="evt-desc">{html.escape(summary)}</div>'
        f"</div>"
    )


def _tool_entry(
    call_id: str,
    name: str,
    args: str,
    result: str | None,
    status: str,
) -> str:
    """One tool-call card for the Tools tab."""
    tid = f"tool-entry-{call_id}"
    ts = _now_str()
    header_label = f"Tool Result ({html.escape(name)})" if result is not None else f"Tool Call ({html.escape(name)})"
    status_badge = (
        '<span class="tool-badge tool-badge-result">RESULT</span>'
        if result is not None
        else '<span class="tool-badge tool-badge-call">CALL</span>'
    )
    entry = (
        f'<div id="{tid}" class="tool-card tool-{status}">'
        f'<div class="tool-hdr">'
        f"<span>⚡ {header_label}</span>"
        f'<span class="evt-ts">{ts}</span>'
        f"</div>"
        f'<div class="tool-detail">{status_badge} <code>{html.escape(name)}</code></div>'
        f'<div class="tool-call-id">Call ID: {html.escape(call_id)}</div>'
    )
    if args:
        esc = html.escape((args[:300] + "…") if len(args) > 300 else args)
        entry += f'<details><summary>Arguments</summary><pre class="tool-pre">{esc}</pre></details>'
    if result is not None:
        esc = html.escape((result[:400] + "…") if len(result) > 400 else result)
        entry += f'<details open><summary>Result</summary><pre class="tool-pre tool-pre-result">{esc}</pre></details>'
    return entry + "</div>"


def _evt_summary(event: Any, etype: str) -> str:
    """Short human-readable summary for an event."""
    if etype == "response.output_text.done":
        t = getattr(event, "text", "")
        return t[:80] if t else "Text complete"
    if etype == "response.output_item.added":
        item = getattr(event, "item", None)
        if item:
            itype = getattr(item, "type", "")
            if itype == "executor_action":
                exec_id = getattr(item, "executor_id", "")
                status = getattr(item, "status", "")
                return f"Executor {exec_id}: {status}"
            if itype == "message":
                content = getattr(item, "content", [])
                if content:
                    text = getattr(content[0], "text", "")[:40]
                    return f"Output: {text}" if text else "Output message"
            name = getattr(item, "name", "")
            if name:
                return f"Tool call: {name}"
        return "Output item added"
    if etype == "response.output_item.done":
        item = getattr(event, "item", None)
        if item:
            itype = getattr(item, "type", "")
            if itype == "executor_action":
                exec_id = getattr(item, "executor_id", "")
                return f"Executor {exec_id}: completed"
            name = getattr(item, "name", "")
            if name:
                return f"Output item done: {name}"
        return "Output item done"
    if etype == "response.workflow_event.completed":
        return "Workflow event"
    if etype == "response.created":
        return "Response created"
    if etype == "response.in_progress":
        return "Response in progress"
    if "function_call" in etype:
        name = getattr(event, "name", "")
        cid = getattr(event, "call_id", "")
        return f'Calling {name}("{cid[:8]}")' if name else f"call {cid[:8]}"
    if etype == "response.function_result.complete":
        out = str(getattr(event, "output", ""))[:60]
        return f"Function result: {out}"
    if etype == "response.completed":
        resp = getattr(event, "response", None)
        usage = getattr(resp, "usage", None) if resp else None
        if usage:
            return f"Response complete ({getattr(usage, 'total_tokens', 0):,} tokens)"
        return "Response complete"
    if etype in ("response.failed", "response.error"):
        return "Error occurred"
    if etype == "response.content_part.added":
        return "Content part added"
    return etype.replace("response.", "")


# region Trace tree builder

_SPAN_BADGE_COLORS: dict[str, str] = {
    "chat": "orange",
    "embeddings": "purple",
    "execute_tool": "blue",
    "create_agent": "green",
    "invoke_agent": "green",
    "workflow.run": "gray",
}

# Operations to show with subdued gray badge instead of colored
_MUTED_OPS = {"message.send", "edge_group.process", "executor.process"}


def _render_span_row(span: dict[str, Any], depth: int = 0) -> str:
    """Render a single span as an expandable row with badge + details."""
    attrs = span.get("attributes", {})
    op = attrs.get("gen_ai.operation.name", span.get("operation_name", "span"))
    name = span.get("operation_name", op)
    duration = span.get("duration_ms")
    inp_tok = attrs.get("gen_ai.usage.input_tokens")
    out_tok = attrs.get("gen_ai.usage.output_tokens")
    tool_name = attrs.get("gen_ai.tool.name", "")
    span_id = span.get("span_id", "")
    trace_id = span.get("trace_id", "")
    status = span.get("status", "")

    # Use muted gray for infrastructure spans (message.send, edge_group, executor.process)
    op_base = name.split(" ")[0] if name else op
    badge_color = "gray" if op_base in _MUTED_OPS else _SPAN_BADGE_COLORS.get(op, "gray")
    # Strip prefixes from display name
    display = tool_name or name or op
    for prefix in ("invoke_agent ", "execute_tool ", "create_agent "):
        if display.startswith(prefix):
            display = display[len(prefix) :]
            break

    dur_html = ""
    if duration is not None:
        dur_html = f'<span class="span-dur">{duration:.1f}ms</span>'

    tok_html = ""
    parts: list[str] = []
    if inp_tok is not None:
        parts.append(f'<span class="tok tok-in">&uarr;{inp_tok}</span>')
    if out_tok is not None:
        parts.append(f'<span class="tok tok-out">&darr;{out_tok}</span>')
    if parts:
        tok_html = '<span class="span-tokens">' + " ".join(parts) + "</span>"

    # Filtered attributes for detail view (skip large message content)
    detail_attrs = {k: v for k, v in attrs.items() if not k.endswith(".messages") and not k.endswith(".message")}
    attrs_json = json.dumps(detail_attrs, indent=2, default=str)

    # Status display
    status_short = str(status).replace("StatusCode.", "")

    indent = depth * 16
    return (
        f'<details class="span-row-wrap" style="padding-left:{indent}px">'
        f'<summary class="span-row">'
        f'<span class="span-chevron">&rsaquo;</span>'
        f'<span class="span-badge badge-{badge_color}">'
        f"{html.escape(display)}</span>"
        f"{dur_html}{tok_html}"
        f"</summary>"
        f'<div class="span-detail">'
        f'<div class="span-detail-row">'
        f'<span class="span-detail-label">Span ID</span>'
        f'<span class="span-detail-val">{html.escape(span_id)}</span></div>'
        f'<div class="span-detail-row">'
        f'<span class="span-detail-label">Trace ID</span>'
        f'<span class="span-detail-val">{html.escape(trace_id)}</span></div>'
        f'<div class="span-detail-row">'
        f'<span class="span-detail-label">Status</span>'
        f'<span class="span-detail-val">{html.escape(status_short)}</span></div>'
        f"<details><summary>Attributes</summary>"
        f'<pre class="span-attrs-pre">{html.escape(attrs_json)}</pre>'
        f"</details>"
        f"</div></details>"
    )


def _build_turn_group(trace_spans: list[dict[str, Any]], turn_id: str) -> str:
    """Build a single turn group with trace spans AND token/composition info."""
    if not trace_spans:
        return ""

    by_id: dict[str, dict[str, Any]] = {}
    for span in trace_spans:
        sid = span.get("span_id", "")
        if sid:
            by_id[sid] = span

    sids = list(by_id.keys())
    sids.sort(key=lambda s: by_id[s].get("start_time") or 0)

    # Root span = longest duration
    root_sid = max(sids, key=lambda s: by_id[s].get("duration_ms") or 0)
    root = by_id[root_sid]

    total_dur = root.get("duration_ms")
    ts = ""
    start_t = root.get("start_time")
    if start_t:
        dt = datetime.fromtimestamp(start_t, tz=timezone.utc).astimezone()
        ts = dt.strftime("%I:%M:%S %p").lstrip("0")

    dur_str = f"{total_dur:.1f}ms" if total_dur is not None else ""

    # Aggregate tokens from non-wrapper spans
    total_input = 0
    total_output = 0
    composition: dict[str, int] = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool_calls": 0,
        "tool_results": 0,
    }
    for span in trace_spans:
        attrs = span.get("attributes", {})
        op = attrs.get("gen_ai.operation.name", "")
        if op in ("invoke_agent", "create_agent"):
            continue
        inp = attrs.get("gen_ai.usage.input_tokens")
        out = attrs.get("gen_ai.usage.output_tokens")
        if inp is not None:
            total_input += int(inp)
        if out is not None:
            total_output += int(out)
        span_comp = _parse_composition(attrs)
        for k in composition:
            composition[k] += span_comp[k]

    total_tokens = total_input + total_output

    # Token bar
    token_bar: str | None = None
    if total_tokens > 0:
        inp_pct = total_input / max(total_tokens, 1) * 100
        out_pct = total_output / max(total_tokens, 1) * 100
        comp_total = sum(composition.values()) or 1
        comp_segs = ""
        for key in _COMP_ORDER:
            count = composition.get(key, 0)
            if count > 0:
                pct = count / comp_total * 100
                color = _COMP_COLORS.get(key, "var(--text-dim)")
                label = _COMP_LABELS.get(key, key)
                comp_segs += (
                    f'<div class="comp-seg" style="width:{pct:.1f}%;'
                    f'background:{color}" title="{label}: ~{count}"></div>'
                )

        token_bar = (
            f'<div class="turn-tokens">'
            f'<div class="ctx-bar">'
            f'<div class="ctx-seg ctx-seg-in" style="width:{inp_pct:.1f}%"></div>'
            f'<div class="ctx-seg ctx-seg-out" style="width:{out_pct:.1f}%"></div>'
            f"</div>"
            f'<div class="ctx-comp-bar">{comp_segs}</div>'
            f'<div class="turn-token-stats">'
            f"{_token_display(total_tokens, total_input, total_output)}"
            f"</div></div>"
        )

    # Token summary for header
    tok_summary = ""
    if total_tokens > 0:
        tok_summary = _token_display(total_tokens, total_input, total_output)

    # Build turn HTML
    turn = (
        f'<details id="turn-{turn_id}" class="turn-group" open>'
        f'<summary class="turn-header">'
        f'<span class="turn-ts">{ts}</span>'
        f'<span class="turn-token-summary">{tok_summary}</span>'
    )
    if dur_str:
        turn += f'<span class="span-dur">{dur_str}</span>'
    turn += f'<span class="turn-count">{len(sids)} spans</span></summary>{token_bar or str()}<div class="turn-spans">'

    # Build parent→children map for tree rendering
    children_map: dict[str, list[str]] = {}
    roots: list[str] = []
    for sid in sids:
        parent = by_id[sid].get("parent_span_id")
        if parent and parent in by_id:
            children_map.setdefault(parent, []).append(sid)
        else:
            roots.append(sid)

    # Second pass: re-parent siblings where one temporally contains another
    # Only re-parent into meaningful containers (invoke_agent contains chat, etc.)
    container_ops = {"invoke_agent", "create_agent", "execute_tool"}
    for pid in list(children_map.keys()):
        siblings = children_map[pid]
        if len(siblings) < 2:
            continue
        # Find container spans among siblings
        reparented: set[str] = set()
        for outer_sid in siblings:
            o = by_id[outer_sid]
            o_op = o.get("attributes", {}).get("gen_ai.operation.name", o.get("operation_name", ""))
            # Only re-parent into known container operations
            if o_op not in container_ops:
                continue
            o_start = o.get("start_time") or 0
            o_end = o.get("end_time") or o_start
            if o_start >= o_end:
                continue
            for inner_sid in siblings:
                if inner_sid == outer_sid or inner_sid in reparented:
                    continue
                inn = by_id[inner_sid]
                i_start = inn.get("start_time") or 0
                i_end = inn.get("end_time") or i_start
                if o_start <= i_start + 0.001 and o_end >= i_end - 0.001:
                    children_map.setdefault(outer_sid, []).append(inner_sid)
                    reparented.add(inner_sid)
        children_map[pid] = [s for s in siblings if s not in reparented]

    # Sort children by start_time
    for pid in children_map:
        children_map[pid].sort(key=lambda s: by_id[s].get("start_time") or 0)

    def _render_tree(sid: str, depth: int) -> str:
        result = _render_span_row(by_id[sid], depth)
        for child_sid in children_map.get(sid, []):
            result += _render_tree(child_sid, depth + 1)
        return result

    for rid in roots:
        turn += _render_tree(rid, 0)

    turn += "</div></details>"
    return turn


def _build_trace_header(turn_count: int, total_spans: int) -> str:
    """Build the trace header bar that gets morphed to update counts."""
    return (
        f'<div id="trace-header" class="trace-header-bar">'
        f'<span class="trace-header-icon">🔍</span>'
        f'<span class="trace-header-title">OTel Spans</span>'
        f'<span class="trace-turn-badge">'
        f"{turn_count} turn{'s' if turn_count != 1 else ''}"
        f" &middot; {total_spans} spans</span>"
        f"</div>"
    )


# region Context Inspector


def _parse_composition(attrs: dict[str, Any]) -> dict[str, int]:
    """Parse token composition from trace span attributes.

    Estimates tokens per message type using character_count / 4,
    matching the React UI's approach. Parses the JSON message arrays
    from ``gen_ai.input.messages`` and ``gen_ai.output.messages``.
    """
    comp: dict[str, int] = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool_calls": 0,
        "tool_results": 0,
    }

    # System instructions (plain string)
    sys_instr = attrs.get("gen_ai.system_instructions", "")
    if sys_instr:
        comp["system"] += len(str(sys_instr)) // 4

    # Parse input/output message arrays
    for msg_key in ("gen_ai.input.messages", "gen_ai.output.messages"):
        raw = attrs.get(msg_key, "")
        if not raw:
            continue
        try:
            messages = json.loads(str(raw)) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            continue
        if not isinstance(messages, list):
            continue

        for msg in messages:
            if not isinstance(msg, dict):
                continue
            role = msg.get("role", "")
            parts = msg.get("parts", [])
            if not isinstance(parts, list):
                continue

            for part in parts:
                if not isinstance(part, dict):
                    continue
                ptype = part.get("type", "")

                if ptype == "text":
                    text = part.get("content", "") or part.get("text", "")
                    chars = len(str(text))
                    if role == "system":
                        comp["system"] += chars // 4
                    elif role == "user":
                        comp["user"] += chars // 4
                    elif role in ("assistant", "model"):
                        comp["assistant"] += chars // 4

                elif ptype == "tool_call":
                    name = part.get("name", "")
                    arguments = part.get("arguments", "")
                    comp["tool_calls"] += (len(str(name)) + len(str(arguments))) // 4

                elif ptype in ("tool_call_response", "tool_result"):
                    result = part.get("response", "") or part.get("result", "") or part.get("output", "")
                    comp["tool_results"] += len(str(result)) // 4

            # Fallback: tool role messages
            if role == "tool":
                for part in parts:
                    if isinstance(part, dict):
                        text = part.get("content", "") or part.get("response", "")
                        comp["tool_results"] += len(str(text)) // 4

    # Also check direct tool attributes
    tool_args = attrs.get("gen_ai.tool.call.arguments", "")
    tool_result = attrs.get("gen_ai.tool.call.result", "")
    if tool_args:
        comp["tool_calls"] += len(str(tool_args)) // 4
    if tool_result:
        comp["tool_results"] += len(str(tool_result)) // 4

    return comp


_COMP_COLORS: dict[str, str] = {
    "system": "var(--primary)",
    "user": "var(--blue)",
    "tool_results": "#f59e0b",
    "tool_calls": "var(--orange)",
    "assistant": "var(--success)",
}

_COMP_LABELS: dict[str, str] = {
    "system": "System",
    "user": "User",
    "tool_results": "Tool results",
    "tool_calls": "Tool calls",
    "assistant": "Assistant",
}

# Input types first, then output types
_COMP_ORDER: list[str] = [
    "system",
    "user",
    "tool_results",  # inputs
    "tool_calls",
    "assistant",  # outputs
]


def _build_context_turn(
    trace_spans: list[dict[str, Any]],
    turn_number: int,
    turn_id: str,
) -> str:
    """Build a Context Inspector turn row with unified token + composition view."""
    if not trace_spans:
        return ""

    total_input = 0
    total_output = 0
    composition: dict[str, int] = {
        "system": 0,
        "user": 0,
        "assistant": 0,
        "tool_calls": 0,
        "tool_results": 0,
    }

    for span in trace_spans:
        attrs = span.get("attributes", {})
        op = attrs.get("gen_ai.operation.name", "")

        # Skip wrapper spans (invoke_agent, create_agent) to avoid double-counting
        # — they aggregate child tokens already
        if op in ("invoke_agent", "create_agent"):
            continue

        inp = attrs.get("gen_ai.usage.input_tokens")
        out = attrs.get("gen_ai.usage.output_tokens")
        if inp is not None:
            total_input += int(inp)
        if out is not None:
            total_output += int(out)

        span_comp = _parse_composition(attrs)
        for k in composition:
            composition[k] += span_comp[k]

    total_tokens = total_input + total_output
    if total_tokens == 0:
        return ""

    # Token bar (proportional, input vs output)
    bar_total = max(total_tokens, 1)
    inp_pct = total_input / bar_total * 100
    out_pct = total_output / bar_total * 100

    # Composition bar (proportional breakdown)
    comp_total = sum(composition.values()) or 1
    comp_segments = ""
    for key in _COMP_ORDER:
        count = composition.get(key, 0)
        if count > 0:
            pct = count / comp_total * 100
            color = _COMP_COLORS.get(key, "var(--text-dim)")
            label = _COMP_LABELS.get(key, key)
            comp_segments += (
                f'<div class="comp-seg" style="width:{pct:.1f}%;'
                f'background:{color}" title="{label}: ~{count} est. tokens ({pct:.0f}%)"></div>'
            )

    # Composition legend (only non-zero entries, in order)
    comp_legend = "".join(
        f'<span class="ctx-leg" style="--c:{_COMP_COLORS.get(k, "gray")}">'
        f"{_COMP_LABELS.get(k, k)}: ~{composition.get(k, 0)}</span>"
        for k in _COMP_ORDER
        if composition.get(k, 0) > 0
    )

    tid = f"ctx-turn-{turn_id}"
    return (
        f'<details id="{tid}" class="ctx-turn">'
        f'<summary class="ctx-turn-row">'
        f'<span class="ctx-num">T{turn_number}</span>'
        f'<div class="ctx-bars">'
        # Token usage bar
        f'<div class="ctx-bar" title="Tokens: ↑{total_input:,} input, ↓{total_output:,} output">'
        f'<div class="ctx-seg ctx-seg-in" style="width:{inp_pct:.1f}%"></div>'
        f'<div class="ctx-seg ctx-seg-out" style="width:{out_pct:.1f}%"></div>'
        f"</div>"
        # Composition estimate bar (thinner, below)
        f'<div class="ctx-comp-bar" title="Estimated composition">{comp_segments}</div>'
        f"</div>"
        f'<span class="ctx-total">'
        f'<span class="tok tok-in">&uarr;{total_input:,}</span> '
        f'<span class="tok tok-out">&darr;{total_output:,}</span>'
        f"</span>"
        f"</summary>"
        # Expandable detail
        f'<div class="ctx-detail">'
        f'<div class="ctx-stat-row">'
        f'<span class="ctx-stat">'
        f'<span class="ctx-label">Input</span> {total_input:,}</span>'
        f'<span class="ctx-stat">'
        f'<span class="ctx-label">Output</span> {total_output:,}</span>'
        f'<span class="ctx-stat">'
        f'<span class="ctx-label">Total</span> {total_tokens:,}</span>'
        f"</div>"
        f'<div class="ctx-comp-section">'
        f'<div class="ctx-comp-label">Estimated composition (~chars/4)</div>'
        f'<div class="ctx-comp-legend">{comp_legend}</div>'
        f"</div></div></details>"
    )


def _build_session_summary(
    total_input: int,
    total_output: int,
    turn_count: int,
) -> str:
    """Build the session summary card for the Context Inspector."""
    total = total_input + total_output
    if total == 0 or turn_count == 0:
        return ""

    avg_total = total // turn_count
    avg_input = total_input // turn_count
    avg_output = total_output // turn_count

    inp_pct = total_input / max(total, 1) * 100
    out_pct = total_output / max(total, 1) * 100

    return (
        f'<div id="ctx-summary" class="ctx-summary">'
        f'<div class="ctx-summary-title">Session Summary</div>'
        f'<div class="ctx-summary-main">'
        f'<div class="ctx-summary-stat">'
        f'<div class="ctx-summary-val">{total:,}</div>'
        f'<div class="ctx-summary-lbl">Total tokens</div></div>'
        f'<div class="ctx-summary-stat">'
        f'<div class="ctx-summary-val ctx-val-in">{total_input:,}</div>'
        f'<div class="ctx-summary-lbl">Input</div></div>'
        f'<div class="ctx-summary-stat">'
        f'<div class="ctx-summary-val ctx-val-out">{total_output:,}</div>'
        f'<div class="ctx-summary-lbl">Output</div></div>'
        f"</div>"
        f'<div class="ctx-bar" style="margin:0.3rem 0">'
        f'<div class="ctx-seg ctx-seg-in" style="width:{inp_pct:.1f}%"></div>'
        f'<div class="ctx-seg ctx-seg-out" style="width:{out_pct:.1f}%"></div>'
        f"</div>"
        f'<div class="ctx-summary-avgs">'
        f"<span>Avg/turn: {avg_total:,}</span>"
        f"<span>Avg input: {avg_input:,}</span>"
        f"<span>Avg output: {avg_output:,}</span>"
        f"<span>Turns: {turn_count}</span>"
        f"</div></div>"
    )


# region Workflow DAG rendering


def _render_dag_node(executor_id: str, class_name: str, *, state: str = "idle") -> str:
    """Render a single DAG node card.

    Args:
        executor_id: The executor identifier.
        class_name: The Python class name of the executor.
        state: Node state — ``idle``, ``running``, or ``completed``.

    Returns:
        HTML string for the DAG node.
    """
    css_class = f"wf-node {state}" if state != "idle" else "wf-node"
    icon = "⚙️"
    if state == "completed":
        icon = "✅"
    elif state == "running":
        icon = "🔄"
    safe_id = html.escape(executor_id)
    safe_class = html.escape(class_name)
    return (
        f'<div id="dag-{safe_id}" class="{css_class}">'
        f'<span class="wf-node-icon">{icon}</span>'
        f'<div class="wf-node-info">'
        f'<span class="wf-node-name">{safe_id}</span>'
        f'<span class="wf-node-class">{safe_class}</span>'
        f"</div></div>"
    )


def build_dag_html(
    executors: list[str],
    edges: list[tuple[str, str]] | None = None,
    *,
    executor_classes: dict[str, str] | None = None,
) -> str:
    """Build the full DAG visualization HTML.

    Args:
        executors: Ordered list of executor IDs.
        edges: Optional edge list ``(source, target)``. Falls back to
            sequential chain when *None*.
        executor_classes: Optional mapping of executor ID to class name.

    Returns:
        HTML string for the DAG container.
    """
    if not executors:
        return '<div id="workflow-dag" class="wf-dag"><div class="sidebar-empty">No executors found</div></div>'

    classes = executor_classes or {}

    # Determine edge order — use provided edges or sequential chain
    ordered: list[str]
    if edges:
        # Topological-ish ordering from edges (BFS from nodes with no incoming)
        incoming: dict[str, set[str]] = {e: set() for e in executors}
        for src, tgt in edges:
            if tgt in incoming:
                incoming[tgt].add(src)
        # Start from nodes with no incoming
        queue = [e for e in executors if not incoming.get(e)]
        if not queue:
            queue = executors[:1]
        visited: set[str] = set()
        ordered = []
        while queue:
            node = queue.pop(0)
            if node in visited:
                continue
            visited.add(node)
            ordered.append(node)
            for src, tgt in edges:
                if src == node and tgt not in visited:
                    queue.append(tgt)
        # Append any missing nodes
        for e in executors:
            if e not in visited:
                ordered.append(e)
    else:
        ordered = list(executors)

    dag_html = ""
    for i, eid in enumerate(ordered):
        dag_html += _render_dag_node(eid, classes.get(eid, "Executor"))
        if i < len(ordered) - 1:
            dag_html += '<div class="wf-edge"></div>'

    return f'<div id="workflow-dag" class="wf-dag">{dag_html}</div>'


def build_workflow_input_form(input_type_name: str, start_executor_id: str) -> str:
    """Build the workflow input form HTML.

    Args:
        input_type_name: The name of the input type for display.
        start_executor_id: The first executor ID.

    Returns:
        HTML string for the input form.
    """
    safe_type = html.escape(input_type_name)
    return (
        f'<div id="workflow-input-form" class="wf-input-bar">'
        f'<div class="wf-input-label">Input ({safe_type}) → {html.escape(start_executor_id)}</div>'
        f'<div class="input-wrap">'
        f'<textarea data-bind:user-input placeholder="Enter workflow input..." rows="2"'
        f" data-on:keydown=\"if (evt.key === 'Enter' && !evt.shiftKey) "
        f"{{ evt.preventDefault(); if ($userInput.trim() !== '' && !$workflowRunning) "
        f"{{ @post('/lite/api/run-workflow') }} }}\""
        f' data-attr:disabled="$workflowRunning"></textarea>'
        f'<button class="send-btn" data-on:click="if ($userInput.trim() !== \'\' && !$workflowRunning) '
        f"{{ @post('/lite/api/run-workflow') }}\""
        f" data-attr:disabled=\"$workflowRunning || $userInput.trim() === ''\">"
        f'<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
        f'<path d="M5 12h14"/><path d="m12 5 7 7-7 7"/></svg>'
        f"</button></div></div>"
    )


def _timeline_item(
    executor_id: str,
    status: str,
    seq: int,
    *,
    duration_ms: float | None = None,
    detail: str = "",
) -> str:
    """Render a single timeline entry for the workflow execution.

    Args:
        executor_id: The executor that was invoked/completed.
        status: One of ``invoked``, ``completed``, or ``error``.
        seq: Sequence number for the timeline item.
        duration_ms: Optional duration in milliseconds.
        detail: Optional additional detail text.

    Returns:
        HTML string for the timeline item.
    """
    icons = {"invoked": "🔄", "completed": "✅", "error": "❌"}
    badge_cls = {"invoked": "badge-purple", "completed": "badge-green", "error": "badge-red"}
    icon = icons.get(status, "⚙️")
    badge = badge_cls.get(status, "badge-gray")
    ts = _now_str()
    safe_id = html.escape(executor_id)
    dur_html = f'<span class="span-dur">{duration_ms:.1f}ms</span>' if duration_ms is not None else ""
    detail_html = f'<span class="wf-tl-detail">{html.escape(detail)}</span>' if detail else ""
    return (
        f'<div id="wf-tl-{seq}" class="wf-timeline-item">'
        f'<span class="wf-tl-icon">{icon}</span>'
        f'<span class="span-badge {badge}">{safe_id}</span>'
        f'<span class="wf-tl-status">{html.escape(status)}</span>'
        f"{dur_html}"
        f'<span class="evt-ts">{ts}</span>'
        f"{detail_html}"
        f"</div>"
    )


def _workflow_complete_card(output_text: str, total_events: int) -> str:
    """Render the final 'Workflow Complete' card.

    Args:
        output_text: The final output text from the workflow.
        total_events: Total number of timeline events.

    Returns:
        HTML string for the completion card.
    """
    rendered = render_markdown(output_text) if output_text else "<em>No output</em>"
    return (
        f'<div id="wf-complete" class="wf-complete">'
        f'<div class="wf-complete-header">'
        f"✅ Workflow Complete"
        f'<span class="evt-ts">{_now_str()}</span>'
        f"</div>"
        f'<div class="wf-complete-body prose">{rendered}</div>'
        f'<div class="wf-complete-stats">{total_events} executor steps</div>'
        f"</div>"
    )


# region Workflow streaming


async def stream_workflow_as_datastar(
    executor: AgentFrameworkExecutor,
    request: AgentFrameworkRequest,
    *,
    executor_ids: list[str] | None = None,
    prev_event_count: int = 0,
    prev_trace_count: int = 0,
) -> AsyncGenerator[str]:
    """Stream workflow execution as Datastar SSE events.

    Handles workflow-specific events like executor invocations and completions,
    emitting timeline items and DAG node state updates.

    Args:
        executor: The Agent Framework executor instance.
        request: The incoming request to execute.
        executor_ids: List of executor IDs for DAG node highlighting.
        prev_event_count: Existing event count from previous runs.
        prev_trace_count: Existing trace span count.

    Yields:
        SSE-formatted strings in Datastar event format.
    """
    seq = 0
    raw_count = 0
    timeline_count = 0
    displayed_event_count = prev_event_count
    trace_count = prev_trace_count
    accumulated_text = ""
    completed_executors: set[str] = set()
    trace_spans: list[dict[str, Any]] = []

    yield _patch_signals({"workflowRunning": True, "error": ""})

    try:
        async for event in executor.execute_streaming(request):
            etype = getattr(event, "type", None)
            raw_count += 1

            # --- Executor action events (from output_item.added/done) ---
            if etype == "response.output_item.added":
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None) if item else None

                if item_type == "executor_action":
                    exec_id = getattr(item, "executor_id", "")
                    status = getattr(item, "status", "")
                    if exec_id and status == "in_progress":
                        timeline_count += 1
                        yield _prepend_elements(
                            "#workflow-timeline",
                            _timeline_item(exec_id, "invoked", timeline_count),
                        )
                        yield _patch_elements(_render_dag_node(exec_id, "Executor", state="running"))
                        yield _patch_signals({"timelineCount": timeline_count})

                elif item_type == "message":
                    # Final output message from workflow
                    content = getattr(item, "content", [])
                    if content:
                        for part in content:
                            text = getattr(part, "text", "")
                            if text:
                                accumulated_text += text

            elif etype == "response.output_item.done":
                item = getattr(event, "item", None)
                item_type = getattr(item, "type", None) if item else None

                if item_type == "executor_action":
                    exec_id = getattr(item, "executor_id", "")
                    status = getattr(item, "status", "")
                    if exec_id and status == "completed":
                        timeline_count += 1
                        completed_executors.add(exec_id)
                        # Extract result from executor
                        result = getattr(item, "result", None)
                        detail = ""
                        if result:
                            detail = str(result[0]) if isinstance(result, list) and result else str(result)
                        yield _prepend_elements(
                            "#workflow-timeline",
                            _timeline_item(exec_id, "completed", timeline_count, detail=detail),
                        )
                        yield _patch_elements(_render_dag_node(exec_id, "Executor", state="completed"))
                        yield _patch_signals({"timelineCount": timeline_count})

            # --- Workflow-level events ---
            elif etype == "response.workflow_event.completed":
                trace_count += 1
                yield _patch_signals({"traceCount": trace_count})

            # --- Trace events (to sidebar) ---
            elif etype == "response.trace.completed":
                trace_count += 1
                trace_data = getattr(event, "data", None)
                if trace_data is not None and not isinstance(trace_data, dict):
                    if hasattr(trace_data, "model_dump"):
                        trace_data = trace_data.model_dump()
                    elif hasattr(trace_data, "__dict__"):
                        trace_data = dict(trace_data.__dict__)
                if isinstance(trace_data, dict) and trace_data.get("span_id"):
                    trace_spans.append(trace_data)
                yield _patch_signals({"traceCount": trace_count})

            # --- Text accumulation ---
            elif etype == "response.output_text.delta":
                accumulated_text += getattr(event, "delta", "")

            elif etype == "response.output_text.done":
                accumulated_text = getattr(event, "text", accumulated_text)

            # --- Errors ---
            elif etype in ("response.failed", "response.error"):
                error_msg = "An error occurred"
                if hasattr(event, "response") and hasattr(event.response, "error"):
                    error_msg = getattr(event.response.error, "message", error_msg)
                elif hasattr(event, "error"):
                    error_msg = getattr(event.error, "message", str(event.error))
                yield _prepend_elements(
                    "#workflow-timeline",
                    _timeline_item("error", "error", timeline_count + 1, detail=error_msg),
                )
                yield _patch_signals({"workflowRunning": False, "error": error_msg})

            # --- Sidebar Events (show all except traces and text deltas) ---
            if etype and etype not in (
                "response.output_text.delta",
                "response.trace.completed",
            ):
                seq += 1
                displayed_event_count += 1
                summary = _evt_summary(event, etype)
                evt_call_id = ""
                yield _prepend_elements(
                    "#events-list",
                    _evt_entry(etype, summary, seq, call_id=evt_call_id),
                )
                pending_signals: dict[str, Any] = {
                    "eventCount": displayed_event_count,
                    "rawCount": raw_count,
                }
                if seq % 5 == 0:
                    yield _patch_signals(pending_signals)

        # --- Stream complete ---
        # Emit response.completed event to sidebar (matches React)
        seq += 1
        displayed_event_count += 1
        raw_count += 1
        yield _prepend_elements(
            "#events-list",
            _evt_entry("response.completed", "Response complete", seq),
        )

        yield _append_elements(
            "#workflow-timeline",
            _workflow_complete_card(accumulated_text, timeline_count),
        )

        # Emit trace turn group (same as agent path)
        if trace_spans:
            turn_id = uuid.uuid4().hex[:8]
            turn_html = _build_turn_group(trace_spans, turn_id)
            if turn_html:
                yield _prepend_elements("#trace-turns", turn_html)

        yield _patch_signals({
            "workflowRunning": False,
            "eventCount": displayed_event_count,
            "rawCount": raw_count,
            "timelineCount": timeline_count,
            "userInput": "",
        })

    except Exception as e:
        logger.error(f"Error in workflow Datastar streaming: {e}", exc_info=True)
        yield _prepend_elements(
            "#workflow-timeline",
            _timeline_item("error", "error", timeline_count + 1, detail=str(e)),
        )
        yield _patch_signals({"workflowRunning": False, "error": str(e)})


# region Agent chat streaming


async def stream_as_datastar(
    executor: AgentFrameworkExecutor,
    request: AgentFrameworkRequest,
    *,
    prev_tool_count: int = 0,
    prev_trace_count: int = 0,
    prev_event_count: int = 0,
    prev_session_input: int = 0,
    prev_session_output: int = 0,
    prev_turn_count: int = 0,
) -> AsyncGenerator[str]:
    """Stream executor events as Datastar SSE events.

    Args:
        executor: The Agent Framework executor instance.
        request: The incoming request to execute.
        prev_tool_count: Existing tool count from previous turns.
        prev_trace_count: Existing trace span count from previous turns.
        prev_event_count: Existing event count from previous turns.
        prev_session_input: Cumulative input tokens from previous turns.
        prev_session_output: Cumulative output tokens from previous turns.
        prev_turn_count: Number of turns completed so far.

    Yields:
        SSE-formatted strings in Datastar event format.
    """
    msg_id = f"msg-{uuid.uuid4().hex[:8]}"
    turn_id = uuid.uuid4().hex[:8]
    # Try to get response_id from request for trace grouping
    extra = getattr(request, "extra_body", None) or {}
    if isinstance(extra, dict) and extra.get("response_id"):
        turn_id = str(extra["response_id"])
    accumulated_text = ""
    seq = 0
    raw_count = 0
    tool_calls: dict[str, dict[str, str]] = {}
    tool_count = prev_tool_count
    turn_tool_count = 0  # Per-turn count for message footer
    trace_count = prev_trace_count
    trace_spans: list[dict[str, Any]] = []
    displayed_event_count = prev_event_count

    yield _patch_signals({"isStreaming": True, "error": ""})

    # Pre-compute avatar HTML (avoid regenerating per-event)
    bot_avatar_html = _bot_avatar()

    # Initial typing indicator
    yield _append_elements(
        "#chat-area",
        f'<div id="{msg_id}" class="msg-row">'
        f"{bot_avatar_html}"
        f'<div class="msg-col">'
        f'<div id="{msg_id}-bubble" class="message assistant-message streaming">'
        f'<span class="typing-indicator">●●●</span>'
        f"</div></div></div>",
    )

    try:
        events: list[Any] = []
        # Throttle: only re-render markdown every N chars to reduce parsing overhead
        last_rendered_len = 0
        render_threshold = 20  # chars between re-renders
        pending_signal_updates: dict[str, Any] = {}

        async for event in executor.execute_streaming(request):
            events.append(event)
            etype = getattr(event, "type", None)
            raw_count += 1

            # --- Sidebar: Traces tab ---
            if etype in (
                "response.trace.completed",
                "response.workflow_event.completed",
            ):
                trace_count += 1
                trace_data = getattr(event, "data", None)
                # Handle both dict and Pydantic model .data
                if trace_data is not None and not isinstance(trace_data, dict):
                    if hasattr(trace_data, "model_dump"):
                        trace_data = trace_data.model_dump()
                    elif hasattr(trace_data, "__dict__"):
                        trace_data = dict(trace_data.__dict__)
                    else:
                        trace_data = None
                if isinstance(trace_data, dict) and trace_data.get("span_id"):
                    trace_spans.append(trace_data)
                pending_signal_updates["traceCount"] = trace_count

            # --- Sidebar Events (skip text deltas and traces) ---
            if etype and etype not in (
                "response.output_text.delta",
                "response.trace.completed",
                "response.workflow_event.completed",
            ):
                seq += 1
                displayed_event_count += 1
                summary = _evt_summary(event, etype)
                # Extract call_id for function events (enables click-to-tool)
                evt_call_id = ""
                if "function" in (etype or ""):
                    evt_call_id = getattr(event, "call_id", "") or getattr(event, "item_id", "") or ""
                yield _prepend_elements(
                    "#events-list",
                    _evt_entry(etype, summary, seq, call_id=evt_call_id),
                )
                # Batch signal updates (emit periodically, not per-event)
                pending_signal_updates["eventCount"] = displayed_event_count
                pending_signal_updates["rawCount"] = raw_count
                if len(pending_signal_updates) >= 3 or displayed_event_count % 5 == 0:
                    yield _patch_signals(pending_signal_updates)
                    pending_signal_updates = {}

            # --- Sidebar Tools ---
            if etype == "response.output_item.added":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    cid = getattr(item, "call_id", uuid.uuid4().hex[:8])
                    fname = getattr(item, "name", "unknown")
                    if cid not in tool_calls:
                        tool_calls[cid] = {
                            "name": fname,
                            "args": "",
                            "result": "",
                            "status": "running",
                        }
                        tool_count += 1
                        turn_tool_count += 1
                        pending_signal_updates["toolCount"] = tool_count
                        yield _prepend_elements(
                            "#tools-list",
                            _tool_entry(cid, fname, "", None, "running"),
                        )

            elif etype == "response.function_call_arguments.delta":
                # item_id contains the call_id for this event type
                cid = getattr(event, "call_id", None) or getattr(event, "item_id", None)
                if cid:
                    if cid not in tool_calls:
                        # Create tool entry if we missed the output_item.added
                        tool_calls[cid] = {
                            "name": "unknown",
                            "args": "",
                            "result": "",
                            "status": "running",
                        }
                        tool_count += 1
                        turn_tool_count += 1
                        pending_signal_updates["toolCount"] = tool_count
                        yield _prepend_elements(
                            "#tools-list",
                            _tool_entry(cid, "unknown", "", None, "running"),
                        )
                    tool_calls[cid]["args"] += getattr(event, "delta", "")

            elif etype == "response.output_item.done":
                item = getattr(event, "item", None)
                if getattr(item, "type", None) == "function_call":
                    cid = getattr(item, "call_id", "")
                    fname = getattr(item, "name", "unknown")
                    # Use item.arguments if available, else our accumulated args
                    item_args = getattr(item, "arguments", "")
                    accumulated_args = tool_calls[cid]["args"] if cid in tool_calls else ""
                    args = item_args or accumulated_args
                    if cid in tool_calls:
                        tool_calls[cid]["args"] = args
                        tool_calls[cid]["name"] = fname
                        tool_calls[cid]["status"] = "done"
                    yield _patch_elements(_tool_entry(cid, fname, args, None, "done"))

            elif etype == "response.function_result.complete":
                cid = getattr(event, "call_id", None)
                result_text = str(getattr(event, "output", ""))
                if cid and cid in tool_calls:
                    tc = tool_calls[cid]
                    tc["result"] = result_text
                    tc["status"] = "done"
                    # Include accumulated args in the final tool card
                    yield _patch_elements(_tool_entry(cid, tc["name"], tc["args"], result_text, "done"))

            # --- Chat: text streaming (throttled markdown rendering) ---
            elif etype == "response.output_text.delta":
                delta = getattr(event, "delta", "")
                accumulated_text += delta
                # Only re-render if enough new text has accumulated
                if len(accumulated_text) - last_rendered_len >= render_threshold:
                    last_rendered_len = len(accumulated_text)
                    rendered = render_markdown(accumulated_text)
                    yield _patch_elements(
                        f'<div id="{msg_id}-bubble" '
                        f'class="message assistant-message streaming">'
                        f'<div class="prose">{rendered}</div>'
                        f"</div>"
                    )

            elif etype == "response.output_text.done":
                final = getattr(event, "text", accumulated_text)
                rendered = render_markdown(final)
                yield _patch_elements(
                    f'<div id="{msg_id}-bubble" '
                    f'class="message assistant-message">'
                    f'<div class="prose">{rendered}</div>'
                    f"</div>"
                )

            # --- Errors ---
            elif etype in ("response.failed", "response.error"):
                error_msg = "An error occurred"
                if hasattr(event, "response") and hasattr(event.response, "error"):
                    error_msg = getattr(event.response.error, "message", error_msg)
                elif hasattr(event, "error"):
                    error_msg = getattr(event.error, "message", str(event.error))
                yield _patch_elements(
                    f'<div id="{msg_id}-bubble" class="message error-message">❌ {html.escape(error_msg)}</div>'
                )
                yield _patch_signals({"isStreaming": False, "error": error_msg})

        # --- Stream complete: final render with token stats ---
        ts = _now_str()
        usage_html = ""
        tool_count_html = f" &bull; 🔧 {turn_tool_count}" if turn_tool_count > 0 else ""

        # Check executor events for response.completed (if emitted by the executor)
        for evt in reversed(events):
            if getattr(evt, "type", None) == "response.completed":
                resp = getattr(evt, "response", None)
                usage = getattr(resp, "usage", None) if resp else None
                if usage:
                    inp = getattr(usage, "input_tokens", 0)
                    out = getattr(usage, "output_tokens", 0)
                    total = getattr(usage, "total_tokens", 0)
                    usage_html = (
                        f'<div class="msg-meta">{ts} &bull; {_token_display(total, inp, out)}{tool_count_html}</div>'
                    )
                break

        # If no completed event found, try to aggregate usage from the executor
        if not usage_html:
            try:
                final_resp = await executor.message_mapper.aggregate_to_response(events, request)
                usage = getattr(final_resp, "usage", None)
                if usage:
                    inp = getattr(usage, "input_tokens", 0)
                    out = getattr(usage, "output_tokens", 0)
                    total = getattr(usage, "total_tokens", 0)
                    usage_html = (
                        f'<div class="msg-meta">{ts} &bull; {_token_display(total, inp, out)}{tool_count_html}</div>'
                    )
                    # Also emit a completed event in the sidebar
                    seq += 1
                    displayed_event_count += 1
                    yield _prepend_elements(
                        "#events-list",
                        _evt_entry(
                            "response.completed",
                            f"Response complete ({total:,} tokens)",
                            seq,
                        ),
                    )
            except Exception:
                logger.debug("Could not aggregate usage from events")

        if not usage_html:
            usage_html = f'<div class="msg-meta">{ts}{tool_count_html}</div>'

        if accumulated_text:
            rendered = render_markdown(accumulated_text)
            yield _patch_elements(
                f'<div id="{msg_id}" class="msg-row">'
                f"{bot_avatar_html}"
                f'<div class="msg-col">'
                f'<div id="{msg_id}-bubble" class="message assistant-message">'
                f'<div class="prose">{rendered}</div>'
                f"</div>"
                f"{usage_html}"
                f"</div></div>"
            )

        yield _patch_signals({
            "isStreaming": False,
            "eventCount": displayed_event_count,
            "rawCount": raw_count,
        })

        # Emit trace turn group and context inspector (prepend so newest is at top)
        if trace_spans:
            turn_html = _build_turn_group(trace_spans, turn_id)
            if turn_html:
                yield _prepend_elements("#trace-turns", turn_html)

            # Session summary: compute this turn's tokens from non-wrapper spans
            turn_input = 0
            turn_output = 0
            for sp in trace_spans:
                sp_attrs = sp.get("attributes", {})
                if sp_attrs.get("gen_ai.operation.name") in ("invoke_agent", "create_agent"):
                    continue
                ti = sp_attrs.get("gen_ai.usage.input_tokens")
                to = sp_attrs.get("gen_ai.usage.output_tokens")
                if ti is not None:
                    turn_input += int(ti)
                if to is not None:
                    turn_output += int(to)

            session_input = prev_session_input + turn_input
            session_output = prev_session_output + turn_output
            session_turns = prev_turn_count + 1

            summary_html = _build_session_summary(
                session_input,
                session_output,
                session_turns,
            )
            if summary_html:
                yield _patch_elements(summary_html)

            yield _patch_signals({
                "sessionInput": session_input,
                "sessionOutput": session_output,
                "turnCount": session_turns,
            })

    except Exception as e:
        logger.error(f"Error in Datastar streaming: {e}", exc_info=True)
        yield _patch_elements(
            f'<div id="{msg_id}-bubble" class="message error-message">❌ Streaming error: {html.escape(str(e))}</div>'
        )
        yield _patch_signals({"isStreaming": False, "error": str(e)})
