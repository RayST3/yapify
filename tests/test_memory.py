"""Test InMemoryHistory adapter."""

from yapwrap.core.memory import InMemoryHistory, ConversationTurn


def test_in_memory_history_basic():
    history = InMemoryHistory(max_turns=3)

    history.add(user_transcript="Hello", agent_response="Hi! How can I help?")
    assert len(history) == 1

    history.add(user_transcript="What time is it?", agent_response="It is noon.")
    assert len(history) == 2

    messages = history.get_messages()
    assert len(messages) == 4
    assert messages[0] == {"role": "user", "content": "Hello"}
    assert messages[1] == {"role": "assistant", "content": "Hi! How can I help?"}


def test_in_memory_history_interruption():
    history = InMemoryHistory()

    history.add(
        user_transcript="Tell me a story",
        agent_response="Once upon a time in a land far far away there lived a king and a queen.",
        was_interrupted=True,
        user_heard="Once upon a time",
    )

    last = history.last_turn()
    assert last is not None
    assert last.was_interrupted is True
    assert last.user_heard == "Once upon a time"

    # Messages should prefer what was actually heard if interrupted
    messages = history.get_messages()
    assert messages[1]["content"] == "Once upon a time"

