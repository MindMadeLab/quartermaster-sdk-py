"""Tests for ContextManager message truncation."""

from quartermaster_engine.messaging.context_manager import ContextManager
from quartermaster_engine.types import Message, MessageRole


def _msg(role: MessageRole, content: str) -> Message:
    return Message(role=role, content=content)


def _sys(content: str) -> Message:
    return _msg(MessageRole.SYSTEM, content)


def _user(content: str) -> Message:
    return _msg(MessageRole.USER, content)


def _asst(content: str) -> Message:
    return _msg(MessageRole.ASSISTANT, content)


class TestTruncateByCount:
    def test_no_truncation_needed(self):
        cm = ContextManager(max_messages=10)
        msgs = [_user("hello"), _asst("hi")]
        result = cm.truncate(msgs)
        assert len(result) == 2

    def test_truncation_preserves_recent(self):
        cm = ContextManager(max_messages=3)
        msgs = [_user("a"), _asst("b"), _user("c"), _asst("d"), _user("e")]
        result = cm.truncate(msgs)
        assert len(result) == 3
        assert result[0].content == "c"
        assert result[1].content == "d"
        assert result[2].content == "e"

    def test_system_message_preserved(self):
        cm = ContextManager(max_messages=3)
        msgs = [_sys("system"), _user("a"), _asst("b"), _user("c"), _asst("d")]
        result = cm.truncate(msgs)
        assert len(result) == 3
        assert result[0].role == MessageRole.SYSTEM
        assert result[1].content == "c"
        assert result[2].content == "d"

    def test_multiple_system_messages(self):
        cm = ContextManager(max_messages=4)
        msgs = [_sys("sys1"), _sys("sys2"), _user("a"), _asst("b"), _user("c")]
        result = cm.truncate(msgs)
        assert len(result) == 4
        assert result[0].content == "sys1"
        assert result[1].content == "sys2"
        assert result[2].content == "b"
        assert result[3].content == "c"

    def test_empty_messages(self):
        cm = ContextManager(max_messages=5)
        assert cm.truncate([]) == []


class TestTruncateByTokens:
    def test_no_truncation_needed(self):
        cm = ContextManager(max_tokens=1000)
        msgs = [_user("hello")]
        result = cm.truncate(msgs)
        assert len(result) == 1

    def test_custom_token_counter(self):
        # Each character = 1 token
        cm = ContextManager(max_tokens=10, token_counter=len)
        msgs = [_user("12345"), _asst("67890"), _user("abcde")]
        result = cm.truncate(msgs)
        # Only the last two fit (5 + 5 = 10)
        assert len(result) == 2
        assert result[0].content == "67890"
        assert result[1].content == "abcde"

    def test_system_message_preserved_in_token_truncation(self):
        cm = ContextManager(max_tokens=15, token_counter=len)
        msgs = [_sys("sys"), _user("aaaa"), _asst("bbbb"), _user("cccc")]
        result = cm.truncate(msgs)
        assert result[0].role == MessageRole.SYSTEM
        # sys=3 tokens, remaining budget=12, cccc=4, bbbb=4, aaaa=4 -> all fit
        assert len(result) == 4

    def test_only_most_recent_messages_kept(self):
        cm = ContextManager(max_tokens=8, token_counter=len)
        msgs = [_user("aaa"), _asst("bbb"), _user("ccc"), _asst("ddd")]
        result = cm.truncate(msgs)
        # Budget 8: ddd=3, ccc=3 -> 6, bbb=3 -> 9 > 8, so keep last 2
        assert len(result) == 2
        assert result[0].content == "ccc"
        assert result[1].content == "ddd"


class TestEstimateTokens:
    def test_default_counter(self):
        cm = ContextManager()
        msgs = [_user("hello world")]
        tokens = cm.estimate_tokens(msgs)
        assert tokens > 0

    def test_custom_counter(self):
        cm = ContextManager(token_counter=lambda t: len(t.split()))
        msgs = [_user("hello world"), _asst("hi there friend")]
        tokens = cm.estimate_tokens(msgs)
        assert tokens == 5  # 2 + 3 words
