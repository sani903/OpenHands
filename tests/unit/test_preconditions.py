import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openhands.controller.agent_controller import AgentController
from openhands.events.action import MessageAction
from openhands.events.event import EventSource
from openhands.llm.preconditions_model import LocalPreConditionsModel

# --- MOCK CLASSES ---

class MockAgent:
    def __init__(self):
        self.llm = MagicMock()
        self.llm.config = MagicMock()
        self.llm.metrics = MagicMock()
        self.config = MagicMock()
        self.name = 'MockAgent'
        self.reset = MagicMock()

    def step(self, state):
        return MagicMock()


class MockEventStream:
    def __init__(self):
        self.events = []
        self.sid = "test-session"

    def add_event(self, event, source):
        self.events.append((event, source))
        return len(self.events)

    def subscribe(self, subscriber, callback, sid):
        pass

    def unsubscribe(self, subscriber, sid):
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
def mock_preconditions_model():
    """Create a mocked preconditions model."""
    model = MagicMock(spec=LocalPreConditionsModel)
    model.generate_checklist = AsyncMock(return_value="1. Test precondition\n2. Another test condition")
    return model


@pytest.fixture
def controller_with_preconditions(mock_preconditions_model):
    """Create an AgentController with a preconditions model."""
    agent = MockAgent()
    event_stream = MockEventStream()
    
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        preconditions_model=mock_preconditions_model,
    )
    
    # Reset for testing
    controller._first_user_message_processed = False
    
    return controller


@pytest.fixture
def controller_without_preconditions():
    """Create an AgentController without a preconditions model."""
    agent = MockAgent()
    event_stream = MockEventStream()
    
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        preconditions_model=None,
    )
    
    # Reset for testing
    controller._first_user_message_processed = False
    
    return controller


# --- TESTS ---

@pytest.mark.asyncio
async def test_handle_message_action_with_preconditions_model(controller_with_preconditions):
    """Verify that when a preconditions model is provided, the first user message is augmented."""
    # Create a user message
    user_message = MessageAction(content='Initial user message')
    user_message._source = EventSource.USER
    
    # Process the message
    await controller_with_preconditions._handle_message_action(user_message)
    
    # Verify the model was called with the right content
    controller_with_preconditions.preconditions_model.generate_checklist.assert_called_once_with('Initial user message')
    
    # Verify the message was augmented
    assert "Test precondition" in user_message.content
    assert "Another test condition" in user_message.content
    assert controller_with_preconditions._first_user_message_processed is True


@pytest.mark.asyncio
async def test_handle_message_action_only_first_message_augmented(controller_with_preconditions):
    """Verify that only the first user message is augmented when multiple messages are processed."""
    # Create two user messages
    message1 = MessageAction(content='First user message')
    message1._source = EventSource.USER
    message2 = MessageAction(content='Second user message')
    message2._source = EventSource.USER

    # Process the first message
    await controller_with_preconditions._handle_message_action(message1)
    
    # Verify the first message was augmented
    assert "Test precondition" in message1.content
    assert controller_with_preconditions._first_user_message_processed is True
    
    # Reset the mock to verify it's not called again
    controller_with_preconditions.preconditions_model.generate_checklist.reset_mock()
    
    # Process the second message
    await controller_with_preconditions._handle_message_action(message2)
    
    # Verify the second message was not augmented
    assert "Test precondition" not in message2.content
    controller_with_preconditions.preconditions_model.generate_checklist.assert_not_called()


@pytest.mark.asyncio
async def test_handle_message_action_non_user_message(controller_with_preconditions):
    """Verify that non-USER messages are not augmented and the flag remains unchanged."""
    # Create a non-user message
    non_user_message = MessageAction(content='Non-user message')
    non_user_message._source = EventSource.AGENT
    
    # Process the message
    await controller_with_preconditions._handle_message_action(non_user_message)
    
    # Verify the message was not augmented
    assert "Test precondition" not in non_user_message.content
    assert controller_with_preconditions._first_user_message_processed is False
    controller_with_preconditions.preconditions_model.generate_checklist.assert_not_called()


def test_augment_task_with_checklist(controller_without_preconditions):
    """Test that the _augment_task_with_checklist helper method produces the expected format."""
    # Create a test message
    original_message = MessageAction(content='Original content')
    original_message._source = EventSource.USER
    
    # Call the helper method
    augmented = controller_without_preconditions._augment_task_with_checklist(
        original_message, 'Sample preconditions'
    )
    
    # Verify the augmented content
    assert 'Original content' in augmented.content
    assert 'Sample preconditions' in augmented.content
    assert '--- Generated Checklist ---' in augmented.content


@pytest.mark.asyncio
async def test_handle_message_action_exception(controller_with_preconditions):
    """Test that if generate_checklist raises an exception, the exception propagates."""
    # Make the model raise an exception
    controller_with_preconditions.preconditions_model.generate_checklist.side_effect = Exception('Test error')
    
    # Create a user message
    user_message = MessageAction(content='Message triggering error')
    user_message._source = EventSource.USER
    
    # Verify the exception propagates
    with pytest.raises(Exception, match='Test error'):
        await controller_with_preconditions._handle_message_action(user_message)
    
    # Verify the flag wasn't set
    assert controller_with_preconditions._first_user_message_processed is False


@pytest.mark.asyncio
async def test_log_is_called_when_handling_message(controller_with_preconditions):
    """Test that the controller's log method is called during message handling."""
    # Mock the log method
    with patch.object(controller_with_preconditions, 'log') as mock_log:
        # Create a user message
        user_message = MessageAction(content='Test log message')
        user_message._source = EventSource.USER
        
        # Process the message
        await controller_with_preconditions._handle_message_action(user_message)
        
        # Verify log was called
        mock_log.assert_called()


@pytest.mark.asyncio
async def test_handle_message_action_no_preconditions_model(controller_without_preconditions):
    """
    Verify that when preconditions_model is None, the message content is not augmented
    and the _first_user_message_processed flag remains False.
    """
    # Create a user message
    user_message = MessageAction(content='Initial user message')
    user_message._source = EventSource.USER
    
    # Process the message
    await controller_without_preconditions._handle_message_action(user_message)
    
    # Verify the message was not augmented
    assert 'Test precondition' not in user_message.content
    assert controller_without_preconditions._first_user_message_processed is False


@pytest.mark.asyncio
async def test_initial_task_is_set(controller_with_preconditions):
    """Test that initial_task is set for the first user message."""
    # Make sure we're testing a fresh controller
    controller_with_preconditions._first_user_message_processed = False
    controller_with_preconditions.initial_task = None
    
    # Create a user message
    user_message = MessageAction(content='Initial task description')
    user_message._source = EventSource.USER
    
    # Process the message
    await controller_with_preconditions._handle_message_action(user_message)
    
    # Verify initial_task was set
    assert controller_with_preconditions.initial_task == 'Initial task description'