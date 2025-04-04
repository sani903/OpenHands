import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from openhands.controller.agent_controller import AgentController
from openhands.core.schema import AgentState
from openhands.events.action import AgentFinishAction, MessageAction
from openhands.events.event import EventSource
from openhands.llm.postconditions_model import LocalPostConditionsModel


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


class MockStuckDetector:
    def __init__(self, state):
        self.state = state
        self.reset = MagicMock()

    def is_stuck(self, headless_mode=True):
        return False


# --- FIXTURES ---
@pytest.fixture
def mock_postconditions_model():
    """Create a mocked postconditions model."""
    model = MagicMock(spec=LocalPostConditionsModel)
    model.generate_checklist = AsyncMock(return_value="1. Test postcondition\n2. Another test condition")
    return model


@pytest.fixture
def controller_with_postconditions(mock_postconditions_model):
    """Create an AgentController with a mocked postconditions model."""
    agent = MockAgent()
    event_stream = MockEventStream()
    
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        postconditions_model=mock_postconditions_model,
    )
    
    # Mock the get_formatted_trajectory_string method
    controller.get_formatted_trajectory_string = MagicMock(return_value="USER (Step 1):\nTest task\n\nASSISTANT (Step 2):\nTest response")
    
    # Set required attributes
    controller.initial_task = "Test task"
    controller.postconditions_passed = False
    controller._initial_max_iterations = 10
    controller.to_refine = False
    controller._stuck_detector = MockStuckDetector(controller.state)
    
    # Mock the set_agent_state_to method
    controller.set_agent_state_to = AsyncMock()
    
    return controller


@pytest.fixture
def controller_with_refinement(mock_postconditions_model):
    """Create an AgentController with a mocked postconditions model and refinement enabled."""
    agent = MockAgent()
    event_stream = MockEventStream()
    
    controller = AgentController(
        agent=agent,
        event_stream=event_stream,
        max_iterations=10,
        postconditions_model=mock_postconditions_model,
    )
    
    # Mock the get_formatted_trajectory_string method
    controller.get_formatted_trajectory_string = MagicMock(return_value="USER (Step 1):\nTest task\n\nASSISTANT (Step 2):\nTest response")
    
    # Set required attributes
    controller.initial_task = "Test task"
    controller.postconditions_passed = False
    controller._initial_max_iterations = 10
    controller.to_refine = True
    controller._stuck_detector = MockStuckDetector(controller.state)
    
    # Mock the set_agent_state_to method
    controller.set_agent_state_to = AsyncMock()
    
    return controller


# --- TESTS ---
@pytest.mark.asyncio
async def test_generate_postconditions(controller_with_postconditions):
    """Test that _generate_postconditions method works correctly."""
    # Call the method
    result = await controller_with_postconditions._generate_postconditions("Test task")
    
    # Verify the postconditions model was called with the right arguments
    controller_with_postconditions.postconditions_model.generate_checklist.assert_called_once_with(
        "Test task", controller_with_postconditions.get_formatted_trajectory_string()
    )
    
    # Verify the result contains the expected checklist
    assert "Test postcondition" in result
    assert "Another test condition" in result


@pytest.mark.asyncio
async def test_handle_finish_action_with_postconditions(controller_with_postconditions):
    """Test that postconditions are generated and a new event is added when agent finishes."""
    # Create an AgentFinishAction
    finish_action = AgentFinishAction(outputs={"result": "Test result"})
    
    # Call the handler
    await controller_with_postconditions._handle_action(finish_action)
    
    # Verify postconditions model was called
    controller_with_postconditions.postconditions_model.generate_checklist.assert_called_once()
    
    # Verify new event was added to stream with the expected content
    assert len(controller_with_postconditions.event_stream.events) == 1
    event, source = controller_with_postconditions.event_stream.events[0]
    
    assert isinstance(event, MessageAction)
    assert "Check how many of the items have been completed" in event.content
    assert "Test postcondition" in event.content
    assert source == EventSource.USER
    
    # Verify postconditions_passed was set to True
    assert controller_with_postconditions.postconditions_passed is True
    
    # Verify agent state was set to RUNNING
    controller_with_postconditions.set_agent_state_to.assert_called_once_with(AgentState.RUNNING)


@pytest.mark.asyncio
async def test_handle_finish_action_with_refinement(controller_with_refinement):
    """Test that refinement prompt is used when to_refine is True."""
    # Create an AgentFinishAction
    finish_action = AgentFinishAction(outputs={"result": "Test result"})
    
    # Call the handler
    await controller_with_refinement._handle_action(finish_action)
    
    # Verify new event was added to stream with refinement prompt
    assert len(controller_with_refinement.event_stream.events) == 1
    event, source = controller_with_refinement.event_stream.events[0]
    
    assert "refine your solution" in event.content
    
    # Verify iterations were extended
    assert controller_with_refinement.state.max_iterations == controller_with_refinement.state.iteration + 10


@pytest.mark.asyncio
async def test_handle_finish_action_postconditions_already_passed(controller_with_postconditions):
    """Test that when postconditions have already passed, the agent finishes normally."""
    # Set postconditions as already passed
    controller_with_postconditions.postconditions_passed = True
    
    # Create an AgentFinishAction
    finish_action = AgentFinishAction(outputs={"result": "Test result"})
    
    # Call the handler
    await controller_with_postconditions._handle_action(finish_action)
    
    # Verify postconditions model was NOT called
    controller_with_postconditions.postconditions_model.generate_checklist.assert_not_called()
    
    # Verify agent state was set to FINISHED
    controller_with_postconditions.set_agent_state_to.assert_called_once_with(AgentState.FINISHED)


@pytest.mark.asyncio
async def test_step_with_max_iterations_and_postconditions(controller_with_postconditions):
    """Test that postconditions are used when max iterations is reached."""
    # Set up the test conditions - we've reached max iterations
    controller_with_postconditions.state.iteration = 10
    controller_with_postconditions.state.agent_state = AgentState.RUNNING
    controller_with_postconditions._handle_traffic_control = AsyncMock(return_value=False)
    
    # Mock _is_stuck to return False
    with patch.object(controller_with_postconditions, '_is_stuck', return_value=False):
        await controller_with_postconditions._step()
    
    # Verify postconditions model was called
    controller_with_postconditions.postconditions_model.generate_checklist.assert_called_once()
    
    # Verify new event was added with maximum steps message
    assert len(controller_with_postconditions.event_stream.events) == 1
    event, source = controller_with_postconditions.event_stream.events[0]
    
    assert "You've reached the maximum number of steps" in event.content


@pytest.mark.asyncio
async def test_step_with_stuck_detection_and_postconditions(controller_with_postconditions):
    """Test that postconditions are used when agent is stuck."""
    # Set the agent state to RUNNING
    controller_with_postconditions.state.agent_state = AgentState.RUNNING
    
    # Mock the is_stuck method to return True
    with patch.object(controller_with_postconditions, '_is_stuck', return_value=True):
        await controller_with_postconditions._step()
    
    # Verify postconditions model was called
    controller_with_postconditions.postconditions_model.generate_checklist.assert_called_once()
    
    # Verify new event was added
    assert len(controller_with_postconditions.event_stream.events) == 1
    
    # Verify stuck detector was reset
    controller_with_postconditions._stuck_detector.reset.assert_called_once()
