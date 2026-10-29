"""
Parser for WhatsApp 'Export chat' .txt files.

Handles the Android and iOS layouts, 12/24-hour clocks, day-first and
month-first dates, RTL/LTR marks, and multi-line messages. Anything that is
not a real user message (encryption notices, joins, deleted messages) is
classified as a system event and excluded from scoring.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime

from .business import UNKNOWN, BusinessContext, detect_account_type

# [12/03/2024, 10:45:12 PM] Sender: text     (iOS, brackets)
IOS_RE = re.compile(
    r"^‎?\[(?P<date>\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap][Mm])?)\]\s*"
    r"(?:‎)?(?P<rest>.*)$"
)
# 12/03/2024, 10:45 pm - Sender: text        (Android, dash)
ANDROID_RE = re.compile(
    r"^‎?(?P<date>\d{1,4}[/.-]\d{1,2}[/.-]\d{2,4}),?\s+"
    r"(?P<time>\d{1,2}:\d{2}(?::\d{2})?(?:\s?[APap]\.?[Mm]\.?)?)\s+[-–]\s+"
    r"(?P<rest>.*)$"
)

SYSTEM_MARKERS = (
    "messages and calls are end-to-end encrypted",
    "created group", "added you", "joined using this group's invite link",
    "changed the subject", "changed this group's icon", "left",
    "removed", "changed their phone number", "you deleted this message",
    "this message was deleted", "missed voice call", "missed video call",
    "security code changed", "disappearing messages", "changed the group description",
    "pinned a message", "turned on admin approval",
)
ATTACHMENT_MARKERS = ("<media omitted>", "image omitted", "video omitted",
                      "audio omitted", "sticker omitted", "document omitted",
                      "gif omitted", "contact card omitted", "this message was edited")

DATE_FORMATS = [
    "%d/%m/%Y", "%m/%d/%Y", "%d/%m/%y", "%m/%d/%y",
    "%Y-%m-%d", "%d-%m-%Y", "%m-%d-%Y", "%d-%m-%y", "%d.%m.%Y", "%d.%m.%y",
]
TIME_FORMATS = ["%I:%M:%S %p", "%I:%M %p", "%H:%M:%S", "%H:%M"]


@dataclass
class ChatMessage:
    index: int
    sender: str | None
    text: str
    timestamp: datetime | None = None
    raw_date: str = ""
    raw_time: str = ""
    is_system: bool = False
    is_attachment: bool = False

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "sender": self.sender,
            "text": self.text,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "is_system": self.is_system,
            "is_attachment": self.is_attachment,
        }


@dataclass
class ParsedChat:
    messages: list[ChatMessage] = field(default_factory=list)
    participants: list[str] = field(default_factory=list)
    unparsed_lines: int = 0
    #: Account standing of the other party, when the export records it.
    business: BusinessContext = field(default_factory=BusinessContext)

    @property
    def account_type(self) -> str:
        return self.business.account_type if self.business else UNKNOWN

    @property
    def user_messages(self) -> list[ChatMessage]:
        return [m for m in self.messages if not m.is_system and m.sender and m.text.strip()]


def _clean(line: str) -> str:
    # Strip the bidi/invisible marks WhatsApp sprinkles into exports.
    return line.replace("‎", "").replace("‏", "").replace("﻿", "").rstrip("\n\r")


def _parse_ts(date_s: str, time_s: str) -> datetime | None:
    time_s = time_s.replace(".", "").replace(" ", " ").strip().upper()
    time_s = re.sub(r"\s*([AP]M)$", r" \1", time_s)
    for df in DATE_FORMATS:
        for tf in TIME_FORMATS:
            try:
                return datetime.strptime(f"{date_s} {time_s}", f"{df} {tf}")
            except ValueError:
                continue
    return None


def _split_sender(rest: str) -> tuple[str | None, str]:
    """`Sender: body` -> ('Sender', 'body'); system lines have no sender."""
    low = rest.lower()
    if any(m in low for m in SYSTEM_MARKERS) and ":" not in rest.split(" ")[0]:
        return None, rest
    m = re.match(r"^(?P<sender>[^:]{1,60}?):\s(?P<body>.*)$", rest, re.DOTALL)
    if not m:
        return None, rest
    return m.group("sender").strip(), m.group("body")


def parse_chat(raw: str) -> ParsedChat:
    chat = ParsedChat()
    current: ChatMessage | None = None
    order: list[str] = []

    for line in raw.splitlines():
        line = _clean(line)
        if not line.strip():
            if current:
                current.text += "\n"
            continue

        match = IOS_RE.match(line) or ANDROID_RE.match(line)
        if not match:
            # Continuation of the previous message.
            if current is not None:
                current.text = (current.text + "\n" + line).strip()
            else:
                chat.unparsed_lines += 1
            continue

        if current is not None:
            chat.messages.append(current)

        rest = match.group("rest")
        sender, body = _split_sender(rest)
        low = body.lower()
        is_system = sender is None or any(m in low for m in SYSTEM_MARKERS)
        current = ChatMessage(
            index=len(chat.messages),
            sender=sender,
            text=body.strip(),
            timestamp=_parse_ts(match.group("date"), match.group("time")),
            raw_date=match.group("date"),
            raw_time=match.group("time"),
            is_system=is_system,
            is_attachment=any(m in low for m in ATTACHMENT_MARKERS),
        )
        if sender is None:
            found = detect_account_type(body)
            # An "official" marker outranks a plain business one.
            if found and (chat.business.account_type == UNKNOWN or found.is_verified):
                chat.business = found
        if sender and not is_system and sender not in order:
            order.append(sender)

    if current is not None:
        chat.messages.append(current)
    chat.participants = order
    return chat


def parse_plain_lines(raw: str) -> ParsedChat:
    """
    Fallback for pasted text with no timestamps.

    Accepts either `Sender: message` per line or one bare message per line.
    """
    chat = ParsedChat()
    order: list[str] = []
    for line in raw.splitlines():
        line = _clean(line).strip()
        if not line:
            continue
        m = re.match(r"^(?P<sender>[^:]{1,40}?):\s(?P<body>.+)$", line)
        sender, body = (m.group("sender").strip(), m.group("body")) if m else (None, line)
        if sender and sender not in order:
            order.append(sender)
        chat.messages.append(ChatMessage(index=len(chat.messages), sender=sender or "Unknown",
                                         text=body.strip()))
    chat.participants = order or ["Unknown"]
    return chat


def parse_any(raw: str) -> ParsedChat:
    """Try the WhatsApp export format, fall back to plain lines."""
    chat = parse_chat(raw)
    if len(chat.user_messages) == 0:
        return parse_plain_lines(raw)
    return chat
