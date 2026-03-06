"""Tests for EventBus — synchronous in-process pub/sub.

Validates subscribe, emit, unsubscribe, clear, thread safety,
error handling, and edge cases.

Target: 12+ tests covering all EventBus methods.
"""

from __future__ import annotations

import threading
import time
from typing import Any

import pytest

from agentassay.persistence.events import EventBus


# ===================================================================
# Basic subscribe / emit
# ===================================================================


class TestSubscribeAndEmit:
    """Tests for the core subscribe + emit flow."""

    def test_single_listener(self) -> None:
        """A single listener receives emitted data."""
        bus = EventBus()
        received: list[dict[str, Any]] = []
        bus.subscribe("trial_complete", lambda d: received.append(d))
        bus.emit("trial_complete", {"trial_num": 1})

        assert len(received) == 1
        assert received[0] == {"trial_num": 1}

    def test_multiple_listeners(self) -> None:
        """All listeners for the same event type receive the data."""
        bus = EventBus()
        results_a: list[dict[str, Any]] = []
        results_b: list[dict[str, Any]] = []
        bus.subscribe("verdict_ready", lambda d: results_a.append(d))
        bus.subscribe("verdict_ready", lambda d: results_b.append(d))

        bus.emit("verdict_ready", {"status": "PASS"})
        assert len(results_a) == 1
        assert len(results_b) == 1

    def test_different_event_types(self) -> None:
        """Listeners only receive events for their subscribed type."""
        bus = EventBus()
        trial_events: list[dict[str, Any]] = []
        run_events: list[dict[str, Any]] = []
        bus.subscribe("trial_complete", lambda d: trial_events.append(d))
        bus.subscribe("run_complete", lambda d: run_events.append(d))

        bus.emit("trial_complete", {"x": 1})
        bus.emit("run_complete", {"y": 2})

        assert len(trial_events) == 1
        assert len(run_events) == 1
        assert trial_events[0] == {"x": 1}
        assert run_events[0] == {"y": 2}

    def test_emit_unknown_event_is_noop(self) -> None:
        """Emitting an event with no subscribers does not raise."""
        bus = EventBus()
        bus.emit("nonexistent_event", {"data": 42})  # Should not raise.

    def test_multiple_emits(self) -> None:
        """Listener is called once per emit."""
        bus = EventBus()
        received: list[dict[str, Any]] = []
        bus.subscribe("sprt_update", lambda d: received.append(d))

        for i in range(5):
            bus.emit("sprt_update", {"step": i})

        assert len(received) == 5


# ===================================================================
# Unsubscribe
# ===================================================================


class TestUnsubscribe:
    """Tests for removing listeners."""

    def test_unsubscribe_removes_listener(self) -> None:
        """After unsubscribe, the listener no longer receives events."""
        bus = EventBus()
        received: list[dict[str, Any]] = []
        handler = lambda d: received.append(d)  # noqa: E731

        bus.subscribe("trial_complete", handler)
        bus.emit("trial_complete", {"a": 1})
        assert len(received) == 1

        bus.unsubscribe("trial_complete", handler)
        bus.emit("trial_complete", {"a": 2})
        assert len(received) == 1  # Still 1 — handler was removed.

    def test_unsubscribe_nonexistent_is_noop(self) -> None:
        """Unsubscribing a handler that was never registered does not raise."""
        bus = EventBus()
        bus.unsubscribe("trial_complete", lambda d: None)  # No-op.

    def test_unsubscribe_only_first_occurrence(self) -> None:
        """If the same handler is subscribed twice, unsubscribe removes one."""
        bus = EventBus()
        received: list[dict[str, Any]] = []
        handler = lambda d: received.append(d)  # noqa: E731

        bus.subscribe("trial_complete", handler)
        bus.subscribe("trial_complete", handler)
        bus.unsubscribe("trial_complete", handler)

        bus.emit("trial_complete", {"a": 1})
        assert len(received) == 1  # One copy remains.


# ===================================================================
# Clear
# ===================================================================


class TestClear:
    """Tests for clearing all subscriptions."""

    def test_clear_removes_all(self) -> None:
        """After clear(), no listeners fire."""
        bus = EventBus()
        received: list[dict[str, Any]] = []
        bus.subscribe("trial_complete", lambda d: received.append(d))
        bus.subscribe("run_complete", lambda d: received.append(d))

        bus.clear()
        bus.emit("trial_complete", {"a": 1})
        bus.emit("run_complete", {"b": 2})

        assert received == []
        assert bus.listener_count == 0


# ===================================================================
# Error handling
# ===================================================================


class TestErrorHandling:
    """Tests for listener exceptions."""

    def test_failing_handler_does_not_block_others(self) -> None:
        """If one handler raises, subsequent handlers still execute."""
        bus = EventBus()
        results: list[str] = []

        def good_handler(d: dict[str, Any]) -> None:
            results.append("ok")

        def bad_handler(d: dict[str, Any]) -> None:
            raise RuntimeError("boom")

        bus.subscribe("trial_complete", bad_handler)
        bus.subscribe("trial_complete", good_handler)

        bus.emit("trial_complete", {})
        assert results == ["ok"]


# ===================================================================
# Thread safety
# ===================================================================


class TestThreadSafety:
    """Basic thread-safety checks for EventBus."""

    def test_concurrent_emit(self) -> None:
        """Multiple threads emitting concurrently do not corrupt state."""
        bus = EventBus()
        received: list[int] = []
        lock = threading.Lock()

        def handler(d: dict[str, Any]) -> None:
            with lock:
                received.append(d["n"])

        bus.subscribe("trial_complete", handler)

        threads = [
            threading.Thread(
                target=bus.emit,
                args=("trial_complete", {"n": i}),
            )
            for i in range(50)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(received) == 50
        assert set(received) == set(range(50))


# ===================================================================
# Utility properties
# ===================================================================


class TestUtilityProperties:
    """Tests for listener_count and event_types."""

    def test_listener_count(self) -> None:
        """listener_count tracks total across event types."""
        bus = EventBus()
        assert bus.listener_count == 0
        bus.subscribe("a", lambda d: None)
        bus.subscribe("b", lambda d: None)
        bus.subscribe("a", lambda d: None)
        assert bus.listener_count == 3

    def test_event_types(self) -> None:
        """event_types returns sorted list of active event types."""
        bus = EventBus()
        bus.subscribe("run_complete", lambda d: None)
        bus.subscribe("trial_complete", lambda d: None)
        assert bus.event_types() == ["run_complete", "trial_complete"]

    def test_event_types_empty_after_clear(self) -> None:
        """event_types returns empty list after clear."""
        bus = EventBus()
        bus.subscribe("x", lambda d: None)
        bus.clear()
        assert bus.event_types() == []
