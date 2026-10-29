"""WhatsApp export parsing across the formats real exports actually use."""

from scamshield.core.chat_parser import parse_any, parse_chat, parse_plain_lines

ANDROID = """12/03/2024, 10:45 pm - Messages and calls are end-to-end encrypted.
12/03/2024, 10:46 pm - +91 98765 43210: Hello sir
13/03/2024, 9:05 am - +91 98765 43210: Please share the OTP
sent to your number
13/03/2024, 9:06 am - Priya: <Media omitted>"""

IOS = """[12/03/2024, 10:46:12 PM] Ravi: Hi there
[12/03/2024, 10:47:02 PM] Anu: hello!"""


def test_android_format():
    chat = parse_chat(ANDROID)
    assert chat.participants == ["+91 98765 43210", "Priya"]
    assert len(chat.user_messages) == 3


def test_system_message_excluded():
    chat = parse_chat(ANDROID)
    assert chat.messages[0].is_system
    assert chat.messages[0] not in chat.user_messages


def test_multiline_message_joined():
    chat = parse_chat(ANDROID)
    otp = [m for m in chat.user_messages if "OTP" in m.text][0]
    assert "sent to your number" in otp.text


def test_attachment_flagged():
    chat = parse_chat(ANDROID)
    assert any(m.is_attachment for m in chat.messages)


def test_ios_format():
    chat = parse_chat(IOS)
    assert chat.participants == ["Ravi", "Anu"]
    assert chat.messages[0].timestamp.hour == 22


def test_timestamps_parsed():
    chat = parse_chat(ANDROID)
    assert all(m.timestamp is not None for m in chat.messages)


def test_plain_lines_fallback():
    chat = parse_plain_lines("Alice: hi\nBob: hello\njust a bare line")
    assert len(chat.messages) == 3
    assert chat.messages[2].sender == "Unknown"


def test_parse_any_falls_back():
    chat = parse_any("Your KYC expired, click here\nwho is this?")
    assert len(chat.user_messages) == 2
