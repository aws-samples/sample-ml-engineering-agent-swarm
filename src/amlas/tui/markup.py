"""Markup-safety helpers for the TUI.

Textual renders notification toasts, and any ``Static``/``Label`` created with the
default ``markup=True``, by running the source string through
``textual.content.Content.from_markup``. That parser treats ``[...]`` tokens as
Rich console markup, so a *dynamic* string that merely happens to contain
square-bracket-shaped tokens — a filesystem path, a ``dict``/``list`` repr, a
lake URI, or a **pydantic-v2 ``ValidationError`` repr**
like::

    Input should be a valid string [type=string_type,
    input_value='data/ai4i/bogie/archive/dataset1.csv', input_type=str]

raises ``textual.markup.MarkupError`` deep in the async message-pump render
cycle — *after* ``notify()`` has already returned — which tears down the whole
TUI.

:func:`escape_markup` is the single, dependency-light helper the TUI uses to make
any untrusted/dynamic string safe to hand to ``notify``/markup-enabled widgets. It
wraps :func:`rich.markup.escape`, which:

* is a **no-op** for text with no ``[`` (so the vast majority of toasts are
  byte-for-byte unchanged), and
* backslash-escapes bracket tokens so ``from_markup`` treats them as literal text
  instead of parsing them as markup. Escaping is **display-transparent**: Rich
  renders ``\\[`` back as ``[``, so the human-readable message is preserved while
  the crash is eliminated.

Kept tiny and Textual-free at import time (only a lazy ``rich.markup`` import) so
it is trivially directly verifiable and safe to call from any layer.
"""

from __future__ import annotations

__all__ = ["escape_markup"]


def escape_markup(text: object) -> str:
    """Escape Rich/Textual console-markup metacharacters in ``text``.

    Returns a string safe to pass to ``App.notify`` / a ``markup=True`` widget:
    bracket-shaped tokens are neutralized so ``Content.from_markup`` never raises
    ``MarkupError`` on dynamic content, while the displayed text is preserved
    (escaping is display-transparent). ``None`` becomes ``""`` and non-``str``
    inputs are coerced via ``str()`` so an error surface never itself raises.
    """
    if text is None:
        return ""
    s = text if isinstance(text, str) else str(text)
    if "[" not in s:
        # Fast path: no markup metacharacter possible → identical string.
        return s
    try:
        from rich.markup import escape

        return escape(s)
    except Exception:
        # Escaping must never be the thing that breaks a notification. Falling
        # back to the raw string keeps the message; the App-level notify override
        # still leaves markup rendering best-effort.
        return s
