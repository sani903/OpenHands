import asyncio
from unittest.mock import patch, AsyncMock, MagicMock

import pytest

from openhands.core.config import AppConfig
from openhands.agents.agent_controller import AgentController
from openhands.llm.local_checklist_model import LocalChecklistModel
from openhands.events.action import MessageAction
from openhands.core.schema import EventSource

# Mock necessary classes and functions
class MockAgent:
    def __init__(self, llm):
        self.llm = llm
        self.config = MagicMock()
        self.name = "MockAgent"

    def step(self, state):
        return MagicMock()  # Replace with a mock Action

class MockEventStream:
    def __init__(self):
        self.events = []

    def add_event(self, event, source):
        self.events.append(event)

    def subscribe(self, subscriber, callback, sid):
        pass

    def get_latest_event_id(self):
        return len(self.events)

    def get_events(self, start_id=0, end_id=None, reverse=False, filter_out_type=None, filter_hidden=True):
        return self.events

@pytest.fixture
def app_config():
    config = AppConfig()
    return config

@pytest.fixture
def agent_controller(app_config):
    agent = MockAgent(llm=None)  # Initialize agent without LLM
    event_stream = MockEventStream()
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        checklist_model=None
    )
    controller._first_user_message_processed = False  # Reset flag
    return controller

@pytest.fixture
def agent_controller_with_checklist_model(app_config):
    agent = MockAgent(llm=None)  # Initialize agent without LLM
    event_stream = MockEventStream()
    
    # Mock LocalChecklistModel
    checklist_model = MagicMock(spec=LocalChecklistModel)
    checklist_model.generate_checklist = AsyncMock(return_value="Sample checklist")
    
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        checklist_model=checklist_model
    )
    controller._first_user_message_processed = False  # Reset flag
    return controller

@pytest.mark.asyncio
async def test_handle_message_action_no_checklist_model(agent_controller):
    # Given
    user_message = MessageAction(content="Initial user message", source=EventSource.USER)

    # When
    await agent_controller._handle_message_action(user_message)

    # Then
    assert "Sample checklist" not in user_message.content
    assert agent_controller._first_user_message_processed is False

@pytest.mark.asyncio
async def test_handle_message_action_with_checklist_model(agent_controller_with_checklist_model):
    # Given
    user_message = MessageAction(content="Initial user message", source=EventSource.USER)

    # When
    await agent_controller_with_checklist_model._handle_message_action(user_message)

    # Then
    assert "Sample checklist" in user_message.content
    assert agent_controller_with_checklist_model._first_user_message_processed is True
