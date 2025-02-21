from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from openhands.controller.agent_controller import AgentController
from openhands.core.config import AppConfig
from openhands.events.action import MessageAction
from openhands.events.event import EventSource
from openhands.llm.checklist_model import LocalChecklistModel

# --- MOCK CLASSES ---


class MockAgent:
    def __init__(self, llm):
        self.llm = llm
        self.config = MagicMock()
        self.name = 'MockAgent'

    def step(self, state):
        return MagicMock()


class MockEventStream:
    def __init__(self):
        self.events = []

    def add_event(self, event, source):
        self.events.append((event, source))

    def subscribe(self, subscriber, callback, sid):
        pass

    def get_latest_event_id(self):
        return len(self.events)

    def get_events(
        self,
        start_id=0,
        end_id=None,
        reverse=False,
        filter_out_type=None,
        filter_hidden=True,
    ):
        return [event for event, src in self.events]


# --- FIXTURES ---


@pytest.fixture
def app_config():
    return AppConfig()


@pytest.fixture
def agent_controller(app_config):
    """
    Create an AgentController without a checklist model.
    In this case, no augmentation should occur.
    """
    agent = MockAgent(llm=None)
    event_stream = MockEventStream()
    controller = AgentController(
        agent=agent, event_stream=event_stream, max_iterations=10, checklist_model=None
    )
    controller._first_user_message_processed = False  # Reset for testing
    return controller


@pytest.fixture
def agent_controller_with_checklist_model(app_config):
    """
    Create an AgentController with a mocked checklist model.
    The mocked generate_checklist returns a fixed value.
    """
    agent = MockAgent(llm=None)
    event_stream = MockEventStream()
    checklist_model = MagicMock(spec=LocalChecklistModel)
    checklist_model.generate_checklist = AsyncMock(return_value='Sample checklist')
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        checklist_model=checklist_model,
    )
    controller._first_user_message_processed = False  # Reset for testing
    return controller


# --- TESTS ---


@pytest.mark.asyncio
async def test_handle_message_action_with_checklist_model(
    agent_controller_with_checklist_model,
):
    """Verify that when a checklist model is provided, the first user message is augmented."""
    user_message = MessageAction(content='Initial user message')
    user_message._source = EventSource.USER
    await agent_controller_with_checklist_model._handle_message_action(user_message)
    assert 'Sample checklist' in user_message.content
    assert agent_controller_with_checklist_model._first_user_message_processed is True


@pytest.mark.asyncio
async def test_handle_message_action_only_first_message_augmented(
    agent_controller_with_checklist_model,
):
    """Verify that only the first user message is augmented when multiple messages are processed."""
    message1 = MessageAction(content='First user message')
    message1._source = EventSource.USER
    message2 = MessageAction(content='Second user message')
    message2._source = EventSource.USER

    await agent_controller_with_checklist_model._handle_message_action(message1)
    assert 'Sample checklist' in message1.content
    assert agent_controller_with_checklist_model._first_user_message_processed is True

    await agent_controller_with_checklist_model._handle_message_action(message2)
    assert 'Sample checklist' not in message2.content


@pytest.mark.asyncio
async def test_handle_message_action_non_user_message(
    agent_controller_with_checklist_model,
):
    """Verify that non-USER messages are not augmented and the flag remains unchanged."""
    non_user_message = MessageAction(content='Non-user message')
    non_user_message._source = EventSource.AGENT
    await agent_controller_with_checklist_model._handle_message_action(non_user_message)
    assert 'Sample checklist' not in non_user_message.content
    assert agent_controller_with_checklist_model._first_user_message_processed is False


def test_augment_task_with_checklist():
    """Test that the _augment_task_with_checklist helper method produces the expected format."""
    original_message = MessageAction(content='Original content')
    original_message._source = EventSource.USER
    # Use a proper event stream so that _init_history does not cause TypeError.
    dummy_event_stream = MockEventStream()
    controller = AgentController(
        agent=MagicMock(),
        event_stream=dummy_event_stream,
        max_iterations=10,
        checklist_model=None,
    )
    augmented = controller._augment_task_with_checklist(
        original_message, 'Sample checklist'
    )
    assert 'Original content' in augmented.content
    assert 'Sample checklist' in augmented.content
    assert '--- Generated Checklist ---' in augmented.content


@pytest.mark.asyncio
async def test_handle_message_action_exception(agent_controller_with_checklist_model):
    """Test that if generate_checklist raises an exception, the exception propagates."""
    agent_controller_with_checklist_model.checklist_model.generate_checklist.side_effect = Exception(
        'Test error'
    )
    user_message = MessageAction(content='Message triggering error')
    user_message._source = EventSource.USER
    with pytest.raises(Exception, match='Test error'):
        await agent_controller_with_checklist_model._handle_message_action(user_message)
    assert agent_controller_with_checklist_model._first_user_message_processed is False


@pytest.mark.asyncio
async def test_log_is_called_when_handling_message(
    agent_controller_with_checklist_model,
):
    """Test that the controller's log method is called during message handling."""
    with patch.object(agent_controller_with_checklist_model, 'log') as mock_log:
        user_message = MessageAction(content='Test log message')
        user_message._source = EventSource.USER
        await agent_controller_with_checklist_model._handle_message_action(user_message)
        mock_log.assert_called()


@pytest.mark.asyncio
async def test_handle_message_action_no_checklist_model(agent_controller):
    """
    Verify that when checklist_model is None, the message content is not augmented
    and the _first_user_message_processed flag remains False.
    """
    user_message = MessageAction(content='Initial user message')
    user_message._source = EventSource.USER
    await agent_controller._handle_message_action(user_message)
    assert 'Sample checklist' not in user_message.content
    assert agent_controller._first_user_message_processed is False
