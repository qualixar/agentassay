# AgentAssay — Token-efficient stochastic testing for AI agents
# Part of Qualixar | Author: Varun Pratap Bhardwaj
# https://qualixar.com | https://varunpratap.com
# License: Apache-2.0

"""Synchronous in-process event bus for AgentAssay.

Provides a lightweight publish/subscribe mechanism so that dashboard
components and reporters can react to lifecycle events without polling
the database.  All callbacks execute synchronously on the emitter's
thread — keep handlers fast or dispatch to a thread pool if needed.

Supported event types (by convention, not enforced):

* ``trial_complete``   — emitted after each trial is saved
* ``verdict_ready``    — emitted after a verdict is computed
* ``sprt_update``      — emitted on SPRT boundary crossings
* ``run_complete``     — emitted when an entire run finishes
* ``coverage_updated`` — emitted after coverage metrics are stored

Thread safety
-------------
Subscription management and emission are guarded by a ``threading.Lock``
so that concurrent trial runners can safely emit events.
"""

from __future__ import annotations

import logging
import threading
from collections import defaultdict
from typing import Any, Callable

logger = logging.getLogger(__name__)

# Type alias for event handlers
EventCallback = Callable[[dict[str, Any]], None]


class EventBus:
    """Synchronous event bus with thread-safe subscribe/emit.

    Examples
    --------
    >>> bus = EventBus()
    >>> received = []
    >>> bus.subscribe("trial_complete", lambda d: received.append(d))
    >>> bus.emit("trial_complete", {"trial_num": 1})
    >>> received
    [{'trial_num': 1}]
    """

    def __init__(self) -> None:
        self._listeners: dict[str, list[EventCallback]] = defaultdict(list)
        self._lock = threading.Lock()

    def subscribe(self, event_type: str, callback: EventCallback) -> None:
        """Register a listener for an event type.

        Parameters
        ----------
        event_type : str
            The event name to listen for (e.g. ``"trial_complete"``).
        callback : EventCallback
            A callable that accepts a single ``dict`` argument.
            Must not raise — exceptions are logged and swallowed.

        Notes
        -----
        Duplicate subscriptions of the same callback are allowed and will
        result in that callback being invoked multiple times per event.
        """
        with self._lock:
            self._listeners[event_type].append(callback)

    def unsubscribe(self, event_type: str, callback: EventCallback) -> None:
        """Remove a previously registered listener.

        Parameters
        ----------
        event_type : str
            The event name.
        callback : EventCallback
            The exact callback object previously passed to ``subscribe``.

        Notes
        -----
        If the callback is not found, this is a no-op (no error raised).
        If the same callback was subscribed multiple times, only the first
        occurrence is removed.
        """
        with self._lock:
            listeners = self._listeners.get(event_type, [])
            try:
                listeners.remove(callback)
            except ValueError:
                pass  # Not subscribed — harmless.

    def emit(self, event_type: str, data: dict[str, Any]) -> None:
        """Fire an event to all registered listeners.

        Parameters
        ----------
        event_type : str
            The event name to fire.
        data : dict[str, Any]
            Payload delivered to each listener.

        Notes
        -----
        Listeners are called synchronously in subscription order.  If a
        listener raises an exception, it is logged at WARNING level and
        does not prevent subsequent listeners from being called.

        Emitting an event type with no subscribers is a no-op.
        """
        with self._lock:
            # Snapshot the listener list so that handlers can safely
            # subscribe/unsubscribe without deadlocking.
            handlers = list(self._listeners.get(event_type, []))

        for handler in handlers:
            try:
                handler(data)
            except Exception:
                logger.warning(
                    "Event handler %r for '%s' raised an exception",
                    handler,
                    event_type,
                    exc_info=True,
                )

    def clear(self) -> None:
        """Remove all subscriptions for all event types."""
        with self._lock:
            self._listeners.clear()

    @property
    def listener_count(self) -> int:
        """Total number of registered listener entries across all event types.

        Returns
        -------
        int
            Sum of listeners across all event types.
        """
        with self._lock:
            return sum(len(v) for v in self._listeners.values())

    def event_types(self) -> list[str]:
        """Return a sorted list of event types that have at least one listener.

        Returns
        -------
        list[str]
            Event type names.
        """
        with self._lock:
            return sorted(
                k for k, v in self._listeners.items() if v
            )
