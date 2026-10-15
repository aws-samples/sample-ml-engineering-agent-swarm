"""Command queue for the chat pane.

The user can issue multiple commands in sequence; instead of a second submission
cancelling the in-flight turn, pending commands **queue** and fire one-after-another
as each completes. A queued
item can be **promoted** to the front ("send immediately" / up-arrow) so a user can
steer a later command ahead of the queue.

This module is intentionally pure — no Textual, no async — so the ordering/promote
semantics are trivially directly verifiable. :class:`~amlas.tui.screens.chat.ChatPane`
owns one :class:`CommandQueue` and drives the UI + worker loop around it.
"""

from __future__ import annotations

import itertools
from collections import deque
from dataclasses import dataclass, field

# Process-wide monotonic counter → stable, unique, human-orderable queue ids.
_id_counter = itertools.count(1)


@dataclass(frozen=True)
class QueuedCommand:
    """One queued chat submission: an immutable (id, text) pair.

    ``id`` is a stable, unique token (used as the DOM id suffix of the item's row
    and its promote button) so the UI can address a specific queued item even after
    others are removed/promoted. ``text`` is the exact submission string (a slash
    command or a natural-language turn), preserved verbatim.
    """

    text: str
    id: str = field(default_factory=lambda: f"q{next(_id_counter)}")


class CommandQueue:
    """FIFO queue of pending chat submissions with front-promotion.

    Semantics:
      * :meth:`enqueue` appends to the back and returns the created
        :class:`QueuedCommand` (so the caller can render/address it).
      * :meth:`dequeue` pops the front item (the next to run) or ``None`` if empty.
      * :meth:`promote` moves the item with the given id to the FRONT so it is the
        next to :meth:`dequeue` — the "send immediately" steering control.
      * :meth:`remove` drops a specific item; :meth:`clear` empties the queue.
      * :attr:`pending` is an immutable snapshot (front → back) for rendering.
    """

    def __init__(self) -> None:
        self._items: deque[QueuedCommand] = deque()

    def enqueue(self, text: str) -> QueuedCommand:
        """Append ``text`` to the back of the queue; return its :class:`QueuedCommand`."""
        item = QueuedCommand(text=text)
        self._items.append(item)
        return item

    def dequeue(self) -> QueuedCommand | None:
        """Pop and return the front item (next to run), or ``None`` if empty."""
        if not self._items:
            return None
        return self._items.popleft()

    def promote(self, item_id: str) -> bool:
        """Move the item with ``item_id`` to the FRONT. Return ``True`` if found.

        Promoting the already-front item is a no-op that still returns ``True``.
        Unknown ids return ``False`` and leave the queue unchanged.
        """
        for idx, item in enumerate(self._items):
            if item.id == item_id:
                if idx != 0:
                    del self._items[idx]
                    self._items.appendleft(item)
                return True
        return False

    def remove(self, item_id: str) -> bool:
        """Drop the item with ``item_id``. Return ``True`` if it was present."""
        for idx, item in enumerate(self._items):
            if item.id == item_id:
                del self._items[idx]
                return True
        return False

    def clear(self) -> None:
        """Empty the queue."""
        self._items.clear()

    @property
    def pending(self) -> tuple[QueuedCommand, ...]:
        """Immutable front→back snapshot of the pending commands (safe to render)."""
        return tuple(self._items)

    @property
    def is_empty(self) -> bool:
        return not self._items

    def __len__(self) -> int:
        return len(self._items)
