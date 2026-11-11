import ast
import re
from collections.abc import Iterable, Mapping


_CONTROL_ESCAPES = {code: f"\\x{code:02x}" for code in range(32)}


def allowed_tag(tag):
    return tag.isascii()


def _read_quoted_text(value):
    return ast.literal_eval(value.translate(_CONTROL_ESCAPES))


def get_category_tags(value, *, allow_normalized=False):
    """Normalize English tags; optionally accept tags with no language prefix."""
    if isinstance(value, Mapping):
        return []
    if isinstance(value, str):
        # CSV stores NumPy arrays as quoted strings without separating commas.
        values = [
            _read_quoted_text(match.group(0))
            for match in re.finditer(
                r"'(?:\\.|[^'\\])*'" r'|"(?:\\.|[^"\\])*"', value
            )
        ]
    elif isinstance(value, Iterable):
        values = value
    else:
        return []

    tags = []
    for value in values:
        if not isinstance(value, str):
            continue
        tag = value.lower().strip()
        if tag.startswith("en:"):
            tag = tag[3:].strip()
        elif not allow_normalized or ":" in tag:
            continue
        if tag and tag != "null":
            tags.append(tag)
    return tags

