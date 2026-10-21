"""Token counting with tiktoken, loaded on first use."""

import json
import logging
import math
from functools import lru_cache
from typing import Any, List, Optional

logger = logging.getLogger("routehub")

_DEFAULT_ENCODING = "o200k_base"

# Fixed per-message and per-reply overheads of the chat format.
_TOKENS_PER_MESSAGE = 3
_TOKENS_PER_NAME = 1
_TOKENS_PER_REPLY = 3


@lru_cache(maxsize=64)
def _encoding_for(model: str) -> Any:
    """Pick the tiktoken encoding for a model, falling back to o200k_base.

    Args:
        model (str): A bare or provider-prefixed model name.

    Returns:
        Any: A tiktoken Encoding, or None when tiktoken cannot load one.
    """
    try:
        import tiktoken
    except ImportError:
        logger.warning(
            "tiktoken is not installed; token counts are estimates."
        )
        return None
    name = model.split("/")[-1] if model else ""
    try:
        return tiktoken.encoding_for_model(name)
    except KeyError:
        pass
    try:
        return tiktoken.get_encoding(_DEFAULT_ENCODING)
    except Exception as error:
        # Raised offline before tiktoken has cached its vocabulary file.
        logger.warning(
            f"Could not load the {_DEFAULT_ENCODING} tokenizer ({error}); "
            "token counts are estimates."
        )
        return None


def _estimate(text: str) -> List[int]:
    """Stand-in token ids sized at roughly four bytes per token.

    Args:
        text (str): The text to measure.

    Returns:
        List[int]: Placeholder ids whose length approximates the count.
    """
    return [0] * math.ceil(len(text.encode("utf-8")) / 4)


def encode(
    model: str = "",
    text: str = "",
    custom_tokenizer: Optional[Any] = None,
) -> List[int]:
    """Tokenize text with the encoding that matches the model.

    Args:
        model (str): Model whose tokenizer to use.
        text (str): The text to tokenize.
        custom_tokenizer (Optional[Any]): An object with an encode method to
            use instead.

    Returns:
        List[int]: Token ids. When no tokenizer can be loaded, placeholder
        ids whose length estimates the count.
    """
    if custom_tokenizer is not None:
        if isinstance(custom_tokenizer, dict):
            custom_tokenizer = custom_tokenizer["tokenizer"]
        return list(custom_tokenizer.encode(text))
    encoding = _encoding_for(model or "gpt-4o")
    if encoding is None:
        return _estimate(text)
    return encoding.encode(text, disallowed_special=())


def decode(
    model: str = "", tokens: Optional[List[int]] = None
) -> str:
    """Turn token ids back into text.

    Args:
        model (str): Model whose tokenizer produced the ids.
        tokens (Optional[List[int]]): The token ids.

    Returns:
        str: The decoded text.
    """
    encoding = _encoding_for(model or "gpt-4o")
    if encoding is None:
        raise RuntimeError("No tokenizer available to decode with.")
    return encoding.decode(tokens or [])


def _content_text(content: Any) -> str:
    """Flatten message content into the text that gets tokenized.

    Args:
        content (Any): A string, a list of content blocks, or None.

    Returns:
        str: The concatenated text.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(block.get("text", ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return str(content)


def token_counter(
    model: str = "",
    text: Optional[str] = None,
    messages: Optional[List[dict]] = None,
) -> int:
    """Count the tokens in a text or a chat message list.

    Args:
        model (str): Model whose tokenizer to use.
        text (Optional[str]): Raw text to count.
        messages (Optional[List[dict]]): Chat messages to count, including the
            chat format's per-message overhead.

    Returns:
        int: The token count.
    """
    if text is not None:
        return len(encode(model, text))
    total = 0
    for message in messages or []:
        if hasattr(message, "model_dump"):
            message = message.model_dump(exclude_none=True)
        total += _TOKENS_PER_MESSAGE
        total += len(
            encode(model, _content_text(message.get("content")))
        )
        for key in ("role", "name", "tool_call_id"):
            if message.get(key):
                total += len(encode(model, message[key]))
        if message.get("name"):
            total += _TOKENS_PER_NAME
        for call in message.get("tool_calls") or []:
            total += len(encode(model, json.dumps(call)))
    return total + (_TOKENS_PER_REPLY if messages else 0)
