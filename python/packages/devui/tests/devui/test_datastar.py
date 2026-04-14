# Copyright (c) Microsoft. All rights reserved.

"""Tests for the Datastar streaming adapter."""

from __future__ import annotations

from typing import Any

from agent_framework_devui._datastar import (
    _patch_elements,
    _patch_signals,
    render_markdown,
    stream_as_datastar,
)

# ---------------------------------------------------------------------------
# Markdown rendering
# ---------------------------------------------------------------------------


class TestRenderMarkdown:
    """Tests for the server-side markdown renderer."""

    def test_renders_basic_paragraph(self) -> None:
        result = render_markdown("Hello world")
        assert "<p>Hello world</p>" in result

    def test_renders_bold_text(self) -> None:
        result = render_markdown("**bold**")
        assert "<strong>bold</strong>" in result

    def test_renders_code_block(self) -> None:
        result = render_markdown("```python\nprint('hi')\n```")
        assert "<code" in result
        assert "print(&#x27;hi&#x27;)" in result or "print('hi')" in result

    def test_renders_inline_code(self) -> None:
        result = render_markdown("Use `foo()` here")
        assert "<code>foo()</code>" in result

    def test_renders_heading(self) -> None:
        result = render_markdown("## Heading Two")
        assert "<h2" in result
        assert "Heading Two" in result

    def test_renders_list(self) -> None:
        result = render_markdown("- item a\n- item b")
        assert "<li>" in result

    def test_renders_table(self) -> None:
        md = "| A | B |\n|---|---|\n| 1 | 2 |"
        result = render_markdown(md)
        assert "<table" in result
        assert "<td" in result

    def test_renders_strikethrough(self) -> None:
        result = render_markdown("~~deleted~~")
        assert "<del>deleted</del>" in result

    def test_empty_string(self) -> None:
        result = render_markdown("")
        assert result == "" or result.strip() == ""


# ---------------------------------------------------------------------------
# Datastar SSE helpers
# ---------------------------------------------------------------------------


class TestPatchElements:
    """Tests for _patch_elements SSE event formatting."""

    def test_single_line_element(self) -> None:
        result = _patch_elements('<div id="foo">bar</div>')
        assert result.startswith("event: datastar-patch-elements\n")
        assert 'data: elements <div id="foo">bar</div>' in result
        assert result.endswith("\n\n")

    def test_multiline_element(self) -> None:
        html = '<div id="foo">\n  <p>hello</p>\n</div>'
        result = _patch_elements(html)
        lines = result.strip().split("\n")
        # First line is the event type
        assert lines[0] == "event: datastar-patch-elements"
        # Each HTML line gets its own data: elements prefix
        assert lines[1] == 'data: elements <div id="foo">'
        assert lines[2] == "data: elements   <p>hello</p>"
        assert lines[3] == "data: elements </div>"


class TestPatchSignals:
    """Tests for _patch_signals SSE event formatting."""

    def test_simple_signals(self) -> None:
        result = _patch_signals({"isStreaming": True, "error": ""})
        assert result.startswith("event: datastar-patch-signals\n")
        assert "isStreaming" in result
        assert result.endswith("\n\n")

    def test_nested_signals(self) -> None:
        result = _patch_signals({"count": 42})
        assert '"count": 42' in result


# ---------------------------------------------------------------------------
# Streaming adapter (integration-level)
# ---------------------------------------------------------------------------


class _FakeTextDeltaEvent:
    """Mimics a response.output_text.delta event."""

    def __init__(self, delta: str, seq: int) -> None:
        self.type = "response.output_text.delta"
        self.delta = delta
        self.sequence_number = seq

    def model_dump_json(self) -> str:
        import json

        return json.dumps({"type": self.type, "delta": self.delta, "sequence_number": self.sequence_number})


class _FakeTextDoneEvent:
    """Mimics a response.output_text.done event."""

    def __init__(self, text: str, seq: int) -> None:
        self.type = "response.output_text.done"
        self.text = text
        self.sequence_number = seq

    def model_dump_json(self) -> str:
        import json

        return json.dumps({"type": self.type, "text": self.text, "sequence_number": self.sequence_number})


class _FakeCompletedEvent:
    """Mimics a response.completed event with usage info."""

    def __init__(self, seq: int) -> None:
        self.type = "response.completed"
        self.sequence_number = seq
        self.response = _FakeResponseWithUsage()

    def model_dump_json(self) -> str:
        import json

        return json.dumps({"type": self.type, "sequence_number": self.sequence_number})


class _FakeResponseWithUsage:
    """Mimics the response object inside completed event."""

    def __init__(self) -> None:
        self.usage = _FakeUsage()


class _FakeUsage:
    def __init__(self) -> None:
        self.total_tokens = 150
        self.input_tokens = 50
        self.output_tokens = 100


class _FakeErrorEvent:
    """Mimics a response.failed event."""

    def __init__(self) -> None:
        self.type = "response.failed"
        self.response = _FakeErrorResponse()

    def model_dump_json(self) -> str:
        import json

        return json.dumps({"type": self.type})


class _FakeErrorResponse:
    def __init__(self) -> None:
        self.error = _FakeErrorDetail()


class _FakeErrorDetail:
    def __init__(self) -> None:
        self.message = "Something went wrong"


class _FakeExecutor:
    """Minimal executor mock for stream_as_datastar tests."""

    def __init__(self, events: list[Any]) -> None:
        self._events = events

    async def execute_streaming(self, request: Any) -> Any:
        for event in self._events:
            yield event


class TestStreamAsDatastar:
    """Tests for the full stream_as_datastar adapter."""

    async def test_streams_text_deltas_as_patch_elements(self) -> None:
        events = [
            _FakeTextDeltaEvent("Hello ", 1),
            _FakeTextDeltaEvent("world!", 2),
            _FakeTextDoneEvent("Hello world!", 3),
            _FakeCompletedEvent(4),
        ]
        executor = _FakeExecutor(events)
        request = object()  # Minimal — adapter only passes it through

        chunks: list[str] = []
        async for chunk in stream_as_datastar(executor, request):  # type: ignore[arg-type]
            chunks.append(chunk)

        combined = "".join(chunks)

        # Should contain patch-elements events with rendered HTML
        assert "event: datastar-patch-elements" in combined
        # Should contain the rendered markdown text
        assert "Hello" in combined
        assert "world" in combined
        # Should signal streaming start and end
        assert '"isStreaming": true' in combined
        assert '"isStreaming": false' in combined

    async def test_streams_error_event(self) -> None:
        events = [_FakeErrorEvent()]
        executor = _FakeExecutor(events)
        request = object()

        chunks: list[str] = []
        async for chunk in stream_as_datastar(executor, request):  # type: ignore[arg-type]
            chunks.append(chunk)

        combined = "".join(chunks)

        # Should contain error message
        assert "Something went wrong" in combined
        assert "error-message" in combined
        # Should stop streaming
        assert '"isStreaming": false' in combined

    async def test_empty_stream(self) -> None:
        """An executor that yields nothing should still signal start/end."""
        executor = _FakeExecutor([])
        request = object()

        chunks: list[str] = []
        async for chunk in stream_as_datastar(executor, request):  # type: ignore[arg-type]
            chunks.append(chunk)

        combined = "".join(chunks)

        assert '"isStreaming": true' in combined
        assert '"isStreaming": false' in combined

    async def test_usage_info_in_final_render(self) -> None:
        events = [
            _FakeTextDeltaEvent("Test", 1),
            _FakeCompletedEvent(2),
        ]
        executor = _FakeExecutor(events)
        request = object()

        chunks: list[str] = []
        async for chunk in stream_as_datastar(executor, request):  # type: ignore[arg-type]
            chunks.append(chunk)

        combined = "".join(chunks)

        # Should include usage information
        assert "150" in combined  # total_tokens
        assert "msg-meta" in combined  # token stats shown in metadata line
