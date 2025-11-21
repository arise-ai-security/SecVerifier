"""This script is used to run the Reproducer, a multi-agent system for reproducing security vulnerabilities.

Usage:
poetry run python main.py --instance-id gpac.cve-2022-3178 --llm-config llm.eval_gpt_4o --iterations 10 --headless --condenser observation_masking
poetry run python main.py --instance-id njs.cve-2022-32414 --llm-config llm.eval_haiku --iterations 5 --headless --condenser recent
poetry run python main.py --llm-config llm.eval_gpt_4o --iterations 10 --headless --limit 5 --condenser llm
"""

import argparse
import asyncio
import dataclasses
import datetime
import json
import os
import re
import time
import traceback
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, cast

import httpx
import pandas as pd
import toml
from datasets import load_dataset
from jinja2 import Environment, FileSystemLoader

from evaluation.utils.shared import (
    EvalException,
    EvalMetadata,
    EvalOutput,
    assert_and_raise,
    get_metrics,
    is_fatal_evaluation_error,
    make_metadata,
    prepare_dataset,
    reset_logger_for_multiprocessing,
    run_evaluation,
)
from openhands.agenthub.codeact_agent.codeact_agent import CodeActAgent
from openhands.controller.agent import Agent
from openhands.controller.state.state import State
from openhands.core.config import (
    AgentConfig,
    OpenHandsConfig as AppConfig,
    SandboxConfig,
    get_llm_config_arg,
    load_openhands_config,
)
from openhands.core.config.condenser_config import (
    CondenserConfig,
    LLMAttentionCondenserConfig,
    LLMSummarizingCondenserConfig,
    ObservationMaskingCondenserConfig,
    RecentEventsCondenserConfig,
)
from openhands.core.logger import openhands_logger as logger
from openhands.core.main import FakeUserResponseFunc, create_runtime, run_controller
from openhands.core.schema import AgentState
from openhands.core.setup import generate_sid
from openhands.events import EventSource
from openhands.events.action import (
    Action,
    AgentDelegateAction,
    AgentFinishAction,
    CmdRunAction,
    MessageAction,
)
from openhands.events.observation import (
    AgentDelegateObservation,
    CmdOutputObservation,
    ErrorObservation,
)
from openhands.events.serialization.event import event_to_dict
from openhands.llm.llm import LLM
from openhands.memory.condenser import Condenser
from openhands.memory.conversation_memory import ConversationMemory
from openhands.runtime.base import Runtime
from openhands.utils.async_utils import call_async_from_sync
from openhands.utils.prompt import PromptManager

# Define encoding fallbacks to try when reading files
ENCODING_FALLBACKS = ['utf-8', 'latin1', 'cp1252', 'iso-8859-1']

# The number of previous events to keep in the condenser
MAX_EVENTS_TO_KEEP = 50

# Define END_STATES if not already imported or defined
END_STATES = [
    AgentState.FINISHED,
    AgentState.REJECTED,
    AgentState.ERROR,
    AgentState.PAUSED,
    AgentState.STOPPED,
]

# Sanitizer error message patterns
SANITIZER_ERROR_PATTERNS = [
    'ERROR: AddressSanitizer:',
    'ERROR: MemorySanitizer:',
    'WARNING: MemorySanitizer:',
    'SUMMARY: UndefinedBehaviorSanitizer:',
    'UndefinedBehaviorSanitizer:DEADLYSIGNAL',
    'ERROR: LeakSanitizer:',
]

# Environment variables to collect
ENV_VARS_TO_COLLECT = [
    'CFLAGS',
    'CXXFLAGS',
]

DELEGATION_SEQUENCE = ['BuilderAgent', 'ExploiterAgent', 'FixerAgent']


@dataclass
class ExecutionResult:
    builder: Dict[str, Any]
    exploiter: Dict[str, Any]
    fixer: Dict[str, Any]

    def to_dict(self) -> Dict[str, Any]:
        return {
            'builder': self.builder,
            'exploiter': self.exploiter,
            'fixer': self.fixer,
        }


@dataclass
class InstanceOutput:
    execution: ExecutionResult
    build_sh: str
    secb_sh: str
    artifacts: Dict[str, str]  # filename -> base64 encoded content
    env: Dict[str, str]  # environment variables like CFLAGS, CXXFLAGS
    base_commit_hash: Optional[str] = None  # Add optional base commit hash
    patch: Optional[str] = None  # Add optional model patch
    repo_changes: Optional[str] = None  # Add optional repository changes diff

    def to_dict(self) -> Dict[str, Any]:
        return {
            'execution': self.execution.to_dict(),
            'build_sh': self.build_sh,
            'secb_sh': self.secb_sh,
            'artifacts': self.artifacts,
            'env': self.env,
            'base_commit_hash': self.base_commit_hash,
            'patch': self.patch,
            'repo_changes': self.repo_changes,
        }


@dataclass
class ReproOutput:
    instance_id: str
    instruction: str
    instance: Dict[str, Any]
    result: InstanceOutput

    def to_dict(self) -> Dict[str, Any]:
        return {
            'instance_id': self.instance_id,
            'instruction': self.instruction,
            'instance': self.instance,
            'result': self.result.to_dict(),
        }


# Define Reproducer that will coordinate the three agents
class Reproducer(CodeActAgent):
    """Reproducer Agent that coordinates vulnerability reproduction across multiple specialized agents.

    This agent orchestrates the process by delegating to:
    1. BuilderAgent - for build script validation and fixing
    2. ExploiterAgent - for helper script improvement and PoC creation
    3. FixerAgent - for finding the correct commit that fixes the vulnerability

    It preserves runtime state across agent delegations to maintain container environment integrity.
    """

    VERSION = '1.0'

    def __init__(
        self,
        llm: LLM,
        config: AgentConfig,
    ) -> None:
        """Initialize the Reproducer.

        Args:
            llm: The LLM to use for this agent
            config: The agent configuration
        """
        super().__init__(llm, config)

        # Setup Jinja2 environment
        self.prompt_template_dir = Path(__file__).parent / 'prompts' / 'instructions'
        self.jinja_env = Environment(
            loader=FileSystemLoader(self.prompt_template_dir), autoescape=True
        )

        # Override the prompt directory to use Reproducer-specific prompts (if needed for other prompts)
        self.prompt_manager = PromptManager(
            prompt_dir=str(Path(__file__).parent / 'prompts' / 'reproducer'),
        )

        # Create a ConversationMemory instance
        self.conversation_memory = ConversationMemory(self.config, self.prompt_manager)

        # Use the condenser from config
        self.condenser = Condenser.from_config(self.config.condenser)
        logger.debug(f'Using condenser: {type(self.condenser)}')

        # Set the delegation sequence
        self.delegation_sequence = DELEGATION_SEQUENCE
        self.current_agent_index = 0

        # Init pending_actions (required for CodeActAgent)
        self.pending_actions: deque[Action] = deque()

        # Store delegate metadata for each agent to help with processing
        self.delegate_metadata: Dict[str, Dict[str, Any]] = {}
        self.delegate_outputs: Dict[str, Dict[str, Any]] = {}
        self.current_delegate_agent_name: Optional[str] = None
        self.instance_info: Dict[str, Any] | None = None

    def _parse_initial_instruction(self, state: State):
        """Parses the instance info from the initial user message."""
        first_user_message = next(
            (
                event
                for event in state.history
                if isinstance(event, MessageAction) and event.source == EventSource.USER
            ),
            None,
        )

        if first_user_message:
            content = first_user_message.content
            # More robust regex to find the JSON block, allowing flexible whitespace
            match = re.search(r'```json\s*(\{.*?\})\s*```', content, re.DOTALL)
            if match:
                json_str = match.group(1)
                try:
                    self.instance_info = json.loads(json_str)
                    if self.instance_info:
                        logger.info(
                            f'Parsed instance info: {list(self.instance_info.keys())}'
                        )
                    else:
                        logger.warning(
                            'Parsed instance info resulted in None or empty dictionary.'
                        )
                    return True
                except json.JSONDecodeError as e:
                    logger.error(f'Failed to parse JSON from initial instruction: {e}')
            else:
                logger.warning(
                    'Could not find JSON block in initial instruction using regex.'
                )
        else:
            logger.warning('Could not find initial user message in history.')
        return False

    def _complete_delegate_agent(self, agent_name: str, outputs: Dict[str, Any]):
        """Processes the outputs from a completed delegate agent and stores them.

        Args:
            agent_name: Name of the delegate agent that completed
            outputs: Dictionary containing the agent's outputs
        """
        logger.info(f'Processing completion of delegate agent: {agent_name}')

        # Store all outputs for potential later use or analysis
        self.delegate_outputs[agent_name] = outputs or {}

        # Extract task completion status from AgentFinishAction
        # The task_completed field should be in the outputs from AgentFinishAction
        task_completed = outputs.get('task_completed', False)
        final_thought = outputs.get('final_thought', '')
        self.delegate_metadata[agent_name] = {
            'task_completed': task_completed,
            'final_thought': final_thought,
            'completion_time': time.time(),
        }

        # output_keys = list(self.delegate_outputs[agent_name].keys())
        # logger.info(f'Stored outputs for {agent_name}: {output_keys}')
        logger.info(f'Task completion status for {agent_name}: {task_completed}')

    def step(self, state: State) -> Action:
        """Determines which agent to delegate to next based on the current state.

        Checks for AgentDelegateObservation to identify delegate completion and task status.
        If task is not completed, re-initiates the same agent.

        Parameters:
            state: The current state object with history and context

        Returns:
            AgentDelegateAction: Delegation to the next appropriate agent
            AgentFinishAction: When all agents have completed their tasks
        """
        # Parse instance info on the first step if not already done
        if self.instance_info is None:
            if not self._parse_initial_instruction(state):
                return AgentFinishAction(
                    thought='Could not parse initial instance information.'
                )

        # Check if we were waiting for a delegate and if it just finished
        last_event = state.history[-1] if state.history else None
        if self.current_delegate_agent_name is not None:
            if isinstance(last_event, AgentDelegateObservation):
                logger.info(
                    f'Detected AgentDelegateObservation for {self.current_delegate_agent_name}.'
                )

                # Process and store the outputs from the observation
                self._complete_delegate_agent(
                    self.current_delegate_agent_name, last_event.outputs
                )

                # Check if the task was completed
                task_completed = last_event.outputs.get('task_completed', 'false')
                final_thought = last_event.outputs.get('final_thought', '')
                if task_completed != 'true':
                    logger.info(
                        f'Task not completed by {self.current_delegate_agent_name}, re-initiating...'
                    )
                    # Don't increment current_agent_index, will retry same agent
                    return self._create_delegate_action(
                        self.current_delegate_agent_name, final_thought
                    )

                self.current_delegate_agent_name = (
                    None  # Reset, ready for the next step
                )
            else:
                # If we are waiting, but the last event isn't the delegate observation, keep waiting.
                logger.info(
                    f'Waiting for delegate {self.current_delegate_agent_name} to complete...'
                )
                return MessageAction(
                    content=f'Waiting for {self.current_delegate_agent_name} to complete its task.'
                )

        # Continue with pending actions if any (should be rare with this structure)
        if self.pending_actions:
            return self.pending_actions.popleft()

        # Check if we're done with all agents
        if self.current_agent_index >= len(self.delegation_sequence):
            logger.info('All agents have completed their tasks. Finishing.')
            return AgentFinishAction(
                thought='I have completed the full vulnerability reproduction sequence by delegating to all required agents.'
            )

        # Get the next agent to delegate to
        next_agent_name = self.delegation_sequence[self.current_agent_index]
        self.current_agent_index += 1
        self.current_delegate_agent_name = next_agent_name

        return self._create_delegate_action(next_agent_name)

    def _create_delegate_action(
        self, agent_name: str, previous_thought: Optional[str] = None
    ) -> Action:
        """Creates a delegate action for the specified agent.

        Args:
            agent_name: Name of the agent to delegate to
            previous_thought: Final thought from previous attempt if the agent is being re-initiated

        Returns:
            Action: Either an AgentDelegateAction or AgentFinishAction
        """
        # Prepare inputs for the agent based on which one it is
        current_inputs = self.instance_info.copy() if self.instance_info else {}
        thought = f'Delegating task to {agent_name}.'

        if agent_name == 'BuilderAgent':
            inputs = current_inputs
            thought = f'I need to delegate to the BuilderAgent first to handle build script validation and fixing for instance {inputs.get("instance_id", "unknown")}.'
            # Load and render BuilderAgent instructions
            template = self.jinja_env.get_template('builder_agent_instruction.j2')
            initial_instruction = template.render(
                instance_id=inputs.get('instance_id', 'unknown'),
                work_dir=inputs.get('work_dir', '/src'),
                base_commit=inputs.get('base_commit', 'Not provided'),
                bug_description=inputs.get('bug_description', 'Not provided'),
            )

        elif agent_name == 'ExploiterAgent':
            builder_outputs = self.delegate_outputs.get('BuilderAgent', {})
            inputs = current_inputs
            logger.info(
                f'Passing builder outputs to ExploiterAgent: {list(builder_outputs.keys())}'
            )

            # NOTE: Check if previous agent (BuilderAgent) has completed
            if (
                'reason' in builder_outputs
                and 'Reached maximum number of steps' in builder_outputs['reason']
            ):
                thought = 'The BuilderAgent has reached the maximum number of steps which means the build script is not valid, so the whole task is finished.'
                logger.warning(thought)
                return AgentFinishAction(thought=thought)

            thought = "Now that the BuilderAgent has completed, I'll delegate to the ExploiterAgent to create the proof-of-concept exploit."
            # Load and render ExploiterAgent instructions
            template = self.jinja_env.get_template('exploiter_agent_instruction.j2')
            initial_instruction = template.render(
                instance_id=inputs.get('instance_id', 'unknown'),
                bug_description=inputs.get('bug_description', 'Not provided'),
                work_dir=inputs.get('work_dir', '/src'),
            )

        elif agent_name == 'FixerAgent':
            exploiter_outputs = self.delegate_outputs.get('ExploiterAgent', {})
            inputs = current_inputs
            logger.info(
                f'Passing exploiter outputs to FixerAgent: {list(exploiter_outputs.keys())}'
            )
            thought = "Now that the ExploiterAgent has completed, I'll delegate to the FixerAgent to create a patch file that fixes the vulnerability based on the candidate fix commits."
            # Load and render FixerAgent instructions
            template = self.jinja_env.get_template('fixer_agent_instruction.j2')
            initial_instruction = template.render(
                instance_id=inputs.get('instance_id', 'unknown'),
                repo=inputs.get('repo', 'unknown'),
                work_dir=inputs.get('work_dir', '/src'),
                bug_description=inputs.get('bug_description', 'Not provided'),
                candidate_fixes='\n'.join(
                    [
                        f'SHA: {commit.get("sha", "N/A")}\nURL: {commit.get("url", "N/A")}'
                        for commit in inputs.get('candidate_fixes', [])
                        if commit.get('url') is not None
                    ]
                ),
            )

        else:
            # Default case or unknown agent
            inputs = current_inputs
            thought = f'Delegating to {agent_name} to continue the vulnerability reproduction process.'
            initial_instruction = f'You are the {agent_name}. Please proceed with your assigned task based on the provided inputs.'

        # Add the previous final thought if provided
        if previous_thought:
            logger.info(
                f'Adding previous final thought from {agent_name}: {previous_thought[:100]}...'
            )
            initial_instruction += f'\n\nYour previous attempt was not completed. Your final thought was:\n<PREVIOUS_THOUGHT>\n{previous_thought}\n</PREVIOUS_THOUGHT>\n\nPlease address the issue and complete the task properly.'

        # IMPORTANT: Use the key 'task' for the initial instruction as expected by AgentController
        delegate_inputs = inputs.copy()
        delegate_inputs['task'] = initial_instruction
        logger.info(f'Issuing delegation to {agent_name}')

        # Create the delegation action
        return AgentDelegateAction(
            agent=agent_name,
            inputs=delegate_inputs,
            thought=thought,
        )


class BuilderAgent(CodeActAgent):
    """BuilderAgent for validating and fixing build scripts."""

    # Override the name property/attribute
    name = 'BuilderAgent'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config)

        # Override the prompt directory to use BuilderAgent-specific prompts (if needed for other prompts)
        self.prompt_manager = PromptManager(
            prompt_dir=str(Path(__file__).parent / 'prompts' / 'builder'),
        )


class ExploiterAgent(CodeActAgent):
    """ExploiterAgent for creating proof-of-concept exploits."""

    name = 'ExploiterAgent'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config)

        # Override the prompt directory to use BuilderAgent-specific prompts (if needed for other prompts)
        self.prompt_manager = PromptManager(
            prompt_dir=str(Path(__file__).parent / 'prompts' / 'exploiter'),
        )


class FixerAgent(CodeActAgent):
    """FixerAgent for finding vulnerability fixes."""

    name = 'FixerAgent'

    def __init__(self, llm: LLM, config: AgentConfig) -> None:
        super().__init__(llm, config)

        # Override the prompt directory to use FixerAgent-specific prompts (if needed for other prompts)
        self.prompt_manager = PromptManager(
            prompt_dir=str(Path(__file__).parent / 'prompts' / 'fixer'),
        )


# Register the agents with OpenHands
Agent.register('Reproducer', Reproducer)
Agent.register('BuilderAgent', BuilderAgent)
Agent.register('ExploiterAgent', ExploiterAgent)
Agent.register('FixerAgent', FixerAgent)


def parse_args():
    parser = argparse.ArgumentParser(
        description='Test the Reproducer system for vulnerability reproduction'
    )
    parser.add_argument(
        '--instance-id',
        type=str,
        help='Vulnerability instance ID (e.g., gpac.cve-2022-3178) for debugging specific instance',
    )
    parser.add_argument(
        '--llm-config',
        type=str,
        default='llm.eval_gpt_4o',
        help='LLM config to use (default: llm.eval_gpt_4o)',
    )
    parser.add_argument(
        '--condenser',
        type=str,
        default='recent',
        help='Condenser to use (default: recent)',
        choices=['recent', 'observation_masking', 'llm', 'llm_attention'],
    )
    parser.add_argument(
        '--iterations',
        type=int,
        default=50,
        help='Maximum number of agent iterations (default: 50)',
    )
    parser.add_argument(
        '--max-budget-per-task',
        type=float,
        default=1.0,
        help='Maximum budget per task (default: 1.0)',
    )
    parser.add_argument(
        '--headless',
        action='store_true',
        help='Run in headless mode (no confirmation required)',
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='./output',
        help='Directory to save outputs (default: ./output)',
    )
    parser.add_argument(
        '--limit',
        type=str,
        help='Limit instances to process. Accepts ranges like ":N" (first N), "M:" (from index M onwards), or "M:N" (from index M to N-1). Indices are 0-based.',
    )
    parser.add_argument(
        '--dataset-name',
        type=str,
        help='Dataset name to use for the run',
        default='SEC-bench/Seed',
    )
    parser.add_argument(
        '--label',
        type=str,
        help='Label to use for the run',
        default='cve',
    )
    parser.add_argument(
        '--num-workers',
        type=int,
        default=1,
        help='Number of workers to use for parallel processing',
    )
    return parser.parse_args()


def filter_dataset(dataset: pd.DataFrame, filter_column: str) -> pd.DataFrame:
    file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.toml')
    logger.info(f'Filtering dataset using {file_path}')
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = toml.load(file)
            logger.info(f'Filtering dataset using {data}')
            if 'selected_ids' in data:
                selected_ids = data['selected_ids']['ids']
                logger.info(
                    f'Filtering {len(selected_ids)} tasks from "selected_ids"...'
                )
                subset = dataset[dataset[filter_column].isin(selected_ids)]
                logger.info(f'Retained {subset.shape[0]} tasks after filtering')
                return subset

    skip_ids = os.environ.get('SKIP_IDS', '').split(',')
    if len(skip_ids) > 0:
        logger.info(f'Filtering {len(skip_ids)} tasks from "SKIP_IDS"...')
        return dataset[~dataset[filter_column].isin(skip_ids)]
    return dataset


def setup_output_dir(instance_id: str, output_dir: str) -> tuple[str, str]:
    """Set up the output directory for the agent run.

    Args:
        instance_id: The vulnerability instance ID
        output_dir: Base directory for output files
    """
    # Create a timestamp for the run
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create the output directory if it doesn't exist
    instance_dir = Path(output_dir) / instance_id / timestamp
    instance_dir.mkdir(parents=True, exist_ok=True)
    completions_dir = instance_dir / 'completions'
    completions_dir.mkdir(exist_ok=True)

    return str(instance_dir), str(completions_dir)


# Define auto-continue function conforming to the protocol
def auto_continue_response(
    state: State,
    encapsulate_solution: bool = False,
    try_parse: Callable[[Action], str] | None = None,
) -> str:
    """Generate auto-continue response when agent asks for input."""
    # Provide a more generic response suitable for any delegate
    msg = (
        'Please continue working on the task if it is not completed.\n'
        'If completed, please move on to the next step or finish the task.\n'
        'If you stuck, please gracefully finish the task.\n'
        'IMPORTANT: YOU SHOULD NEVER ASK FOR HUMAN HELP.\n'
    )

    if state.history:
        # check if the last action has an answer, if so, early exit
        if try_parse is not None:
            last_action = next(
                (
                    event
                    for event in reversed(state.history)
                    if isinstance(event, Action)
                ),
                None,
            )
            ans = try_parse(last_action)
            if ans is not None:
                return '/exit'

        # check if the agent has tried to talk to the user 3 times, if so, let the agent know it can give up
        user_msgs = [
            event
            for event in state.history
            if isinstance(event, MessageAction) and event.source == 'user'
        ]
        if len(user_msgs) >= 3:
            # let the agent know that it can give up when it has tried 3 times
            return (
                msg
                + 'If you want to give up, use the "finish" tool to finish the interaction.\n'
            )
    return msg


def process_instance(
    instance: pd.Series,
    metadata: EvalMetadata,
    reset_logger: bool = True,
    runtime_failure_count: int = 0,
) -> EvalOutput:
    """Process a single instance of the dataset.

    Args:
        instance: The instance to process
        metadata: Metadata for the evaluation run
        reset_logger: Whether to reset the logger for this instance
        runtime_failure_count: Number of previous runtime failures (for retries)

    Returns:
        EvalOutput: The evaluation output
    """
    instance_id = instance['instance_id']
    run_output_dir, completions_dir = setup_output_dir(
        instance_id, metadata.eval_output_dir
    )

    # Configure logging based on instance
    if reset_logger:
        log_dir = os.path.join(metadata.eval_output_dir, 'infer_logs')
        reset_logger_for_multiprocessing(logger, instance_id, log_dir)

    # Get configuration parameters from metadata
    headless = metadata.details.get('headless', False) if metadata.details else False
    llm_config_arg = (
        metadata.details.get('llm_config_arg', 'llm.eval_gpt_4o')
        if metadata.details
        else 'llm.eval_gpt_4o'
    )
    condenser_type = (
        metadata.details.get('condenser_type', 'recent')
        if metadata.details
        else 'recent'
    )
    condenser_config = (
        metadata.condenser_config.type if metadata.condenser_config else 'noop'
    )
    max_iterations = metadata.max_iterations
    max_budget_per_task = (
        metadata.details.get('max_budget_per_task', 1.0) if metadata.details else 1.0
    )

    runtime_container_image = f'hwiwonlee/secb.x86_64.{instance_id}:latest'

    logger.info(f'Processing instance: {instance_id} using run_controller')
    logger.info(f'Using runtime container image: {runtime_container_image}')
    logger.info(f'Using LLM config: {llm_config_arg}')
    logger.info(f'Using Condenser: {condenser_type}')
    logger.info(f'Using Condenser config: {condenser_config}')
    logger.info(f'Max iterations: {max_iterations}')
    logger.info(f'Headless mode: {headless}')
    logger.info(f'Trajectory directory: {run_output_dir}')
    logger.info(f'Log completions folder: {completions_dir}')
    logger.info('-' * 50)

    runtime = None
    result = None
    task_state = None  # Renamed from state to avoid name conflict
    repro_output = None
    try:
        # Initialize runtime to avoid UnboundLocalError

        # Configure the LLM - use the config from metadata which already has fallback applied
        llm_config = metadata.llm_config
        if llm_config is None:
            raise EvalException(f"Could not load LLM config from metadata")

        llm_config.log_completions = True
        llm_config.log_completions_folder = completions_dir

        # Configure the default agent (Reproducer) using the selected condenser
        agent_config = AgentConfig(
            # condenser=condenser_config,
            condenser=metadata.condenser_config,
            enable_browsing=True,
            enable_jupyter=True,
            enable_llm_editor=False,
            enable_prompt_extensions=False,
            enable_history_truncation=True,
            disabled_microagents=['github', 'security', 'docker', 'lint'],
        )

        # Configure sandbox
        sandbox_config = SandboxConfig(
            enable_auto_lint=False,
            use_host_network=True,
            platform='linux/amd64',
            timeout=600,  # 10 minutes for overall reproduction
            user_id=0,
            runtime_container_image=runtime_container_image,
            runtime_startup_env_vars={
                'NO_CHANGE_TIMEOUT_SECONDS': '300'
            },  # Set to ensure that build commands are completed
            docker_runtime_kwargs={
                'auto_remove': True,
            },
        )

        # Create the AppConfig
        app_config = AppConfig(
            agents={
                'Reproducer': agent_config,
                'BuilderAgent': agent_config,
                'ExploiterAgent': agent_config,
                'FixerAgent': agent_config,
            },
            default_agent='Reproducer',  # Set Reproducer as the default agent
            max_iterations=max_iterations,
            max_budget_per_task=max_budget_per_task,
            runtime='docker',
            sandbox=sandbox_config,
            workspace_base=None,  # Let run_controller handle this if needed
            workspace_mount_path=None,
            run_as_openhands=False,
            # Set path for trajectory saving
            save_trajectory_path=run_output_dir,
            # Optionally disable screenshot saving if not needed
            save_screenshots_in_trajectory=False,
        )
        app_config.set_llm_config(llm_config)
        app_config.set_agent_config(agent_config)

        instance_info = {
            'instance_id': instance_id,
            'repo': instance.get('repo', 'Not provided'),
            'base_commit': instance.get('base_commit', 'Not provided'),
            'work_dir': instance.get('work_dir', '/src'),
            'build_sh': instance.get('build_sh', 'Not provided'),
            'bug_description': instance.get('bug_description', 'Not provided'),
            # Convert ndarray to list if necessary for JSON serialization
            'candidate_fixes': (
                instance.get('candidate_fixes', []).tolist()
                if hasattr(instance.get('candidate_fixes', []), 'tolist')
                else instance.get('candidate_fixes', [])
            ),
        }
        instruction = f"""
Please coordinate the vulnerability reproduction process for the following instance:
```json
{json.dumps(instance_info, indent=2)}
```
I will delegate to specialized agents sequentially: {', '.join(DELEGATION_SEQUENCE)}.
Please start by delegating to the {DELEGATION_SEQUENCE[0]}.
"""

        initial_message = MessageAction(content=instruction)
        sid = generate_sid(app_config)

        runtime = create_runtime(app_config)
        call_async_from_sync(runtime.connect)

        try:
            fake_user_resp_fn = (
                cast(FakeUserResponseFunc, auto_continue_response) if headless else None
            )
            task_state: State | None = asyncio.run(
                run_controller(
                    config=app_config,
                    initial_user_action=initial_message,
                    sid=sid,
                    runtime=runtime,
                    fake_user_response_fn=fake_user_resp_fn,
                    headless_mode=headless,
                )
            )

            if task_state:
                # if fatal error, throw Exception to trigger re-run
                if is_fatal_evaluation_error(task_state.last_error):
                    raise EvalException(
                        'Fatal error detected: ' + task_state.last_error
                    )

                logger.info(
                    f'Task completed successfully for {instance_id}: {task_state.agent_state}'
                )
                # Get results from the runtime - properly handle the async function
                result_coroutine = complete_runtime(runtime, instance)
                result = asyncio.run(result_coroutine)  # Properly await the coroutine

                if result:
                    # Construct the final ReproOutput
                    repro_output = ReproOutput(
                        instance_id=instance_id,
                        instruction=instruction,
                        instance=instance_info,
                        result=result,
                    )

                    # Save the ReproOutput to JSON
                    output_file_path = Path(run_output_dir) / 'output.json'
                    try:
                        with open(output_file_path, 'w') as f:
                            json.dump(repro_output.to_dict(), f, indent=2)
                        logger.info(f'Saved ReproOutput to {output_file_path}')
                    except Exception as json_err:
                        logger.error(f'Failed to save ReproOutput to JSON: {json_err}')
                else:
                    logger.warning(
                        'Could not retrieve results from runtime completion.'
                    )
            else:
                raise EvalException('run_controller did not return a final state.')
        except Exception as e:
            if isinstance(e, httpx.ReadTimeout) or 'ReadTimeout' in str(e):
                logger.error(f'HTTP read timeout occurred: {str(e)}')
                logger.error(
                    'This is likely due to network issues or the LLM service being slow/unavailable.'
                )
                logger.info('Will continue with next instance if available.')
                raise EvalException(f'HTTP read timeout: {str(e)}')
            else:
                error_msg = str(e)
                logger.error(f'Error processing instance {instance_id}: {error_msg}')
                logger.error(traceback.format_exc())
                raise EvalException(f'Error processing instance: {error_msg}')

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error processing instance {instance_id}: {error_msg}')
        logger.error(traceback.format_exc())
        if isinstance(e, EvalException):
            raise  # Re-raise EvalException to trigger retry
        else:
            raise EvalException(f'Unhandled error: {error_msg}')
    finally:
        if runtime:
            runtime.close()  # Ensure runtime is closed asynchronously
            logger.info(f'Runtime for instance {instance_id} closed.')

    if task_state is None:
        raise ValueError('State should not be None.')

    # Convert the history to a serializable format
    histories = [event_to_dict(event) for event in task_state.history]
    metrics = get_metrics(task_state)

    # Create and return the EvalOutput with properly serialized result
    result_dict = {}
    if result:
        try:
            result_dict = dataclasses.asdict(result)
        except TypeError:
            # If result is not a dataclass, convert it manually
            logger.warning('Result is not a dataclass, serializing manually')
            result_dict = {
                'execution': {
                    'builder': getattr(result, 'execution', {}).get('builder', {}),
                    'exploiter': getattr(result, 'execution', {}).get('exploiter', {}),
                    'fixer': getattr(result, 'execution', {}).get('fixer', {}),
                },
                'build_sh': getattr(result, 'build_sh', ''),
                'secb_sh': getattr(result, 'secb_sh', ''),
                'artifacts': getattr(result, 'artifacts', {}),
                'env': getattr(result, 'env', {}),
                'base_commit_hash': getattr(result, 'base_commit_hash', None),
                'patch': getattr(result, 'patch', None),
                'repo_changes': getattr(result, 'repo_changes', None),
            }

    output = EvalOutput(
        instance_id=instance_id,
        instruction=instruction,
        instance=instance_info,
        test_result=result_dict,
        metadata=metadata,
        history=histories,
        metrics=metrics,
        error=task_state.last_error if task_state and task_state.last_error else None,
    )

    return output


async def complete_runtime(
    runtime: Runtime, instance: pd.Series
) -> InstanceOutput | None:
    """Complete the runtime for the agent."""
    logger.info('-' * 30)
    logger.info('BEGIN Runtime Completion Fn')
    logger.info('-' * 30)
    obs: CmdOutputObservation

    # Initialize variables to store the results
    build_script = None
    secb_script = None
    artifacts: Dict[str, str] = {}
    env: Dict[str, str] = {}
    base_commit_hash: Optional[str] = None
    patch: Optional[str] = None
    repo_changes: Optional[str] = None

    # Liveness check
    action = CmdRunAction(command='pwd')
    action.set_hard_timeout(30)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if obs.exit_code != 0:
        # The previous command is still running
        # We need to kill previous command
        logger.info('The previous command is still running, trying to kill it...')
        action = CmdRunAction(command='C-c')
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        # Then run the command again
        action = CmdRunAction(command='pwd')
        action.set_hard_timeout(180)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    assert_and_raise(
        isinstance(obs, CmdOutputObservation) and obs.exit_code == 0,
        f'Failed to run the command: {str(action.command)}',
    )

    # Extract build script from the runtime
    action = CmdRunAction(command="""cat /src/build.sh""")
    action.set_hard_timeout(30)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to extract build script: {str(obs)}')

    if isinstance(obs, CmdOutputObservation):
        build_script = obs.content.strip()
    else:
        assert_and_raise(False, f'Unexpected observation type: {str(obs)}')

    # Extract environment variables
    for env_var in ENV_VARS_TO_COLLECT:
        action = CmdRunAction(command=f"""printenv {env_var} || echo """)
        action.set_hard_timeout(30)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
            value = obs.content.strip()
            if value:  # Only add non-empty values
                env[env_var] = value
                logger.info(f'Extracted {env_var}={value}')
        elif isinstance(obs, ErrorObservation):
            logger.error(f'Error extracting {env_var}: {obs.error_id}')

    # Extract secb script from the runtime
    action = CmdRunAction(command="""cat /usr/local/bin/secb""")
    action.set_hard_timeout(30)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    assert_and_raise(obs.exit_code == 0, f'Failed to extract secb script: {str(obs)}')

    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        secb_script = obs.content.strip()
    elif isinstance(obs, CmdOutputObservation):
        logger.warning(
            f'Failed to extract secb script (exit code {obs.exit_code}):\n{obs.content}'
        )
        # Allow continuation even if secb script extraction fails
    elif isinstance(obs, ErrorObservation):
        logger.error(f'Error extracting secb script: {obs.error_id}')
        return None

    # Listing artifacts from the runtime
    action = CmdRunAction(
        # command="""find /testcase -maxdepth 1 -type f ! -name "base_commit_hash" ! -name "model_patch.diff" ! -name "repo_changes.diff" -printf '%f\n'"""
        command="""find /testcase -maxdepth 2 -type f -printf '%P\n'"""
    )
    action.set_hard_timeout(30)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if isinstance(obs, CmdOutputObservation):
        if obs.exit_code == 0 and obs.content:
            file_paths = [line for line in obs.content.strip().split('\n') if line]
            # logger.debug(f'Found files: {file_paths}')

            # Read and base64 encode each file
            for file_path in file_paths:
                # Basic check to avoid issues with filenames containing special characters
                if '..' in file_path:
                    logger.warning(
                        f'Skipping potentially problematic file path: {file_path}'
                    )
                    continue

                # Use binary mode to avoid encoding issues
                read_cmd = f'cat /testcase/{file_path} | base64 -w 0'
                read_action = CmdRunAction(command=read_cmd)
                read_action.set_hard_timeout(30)
                logger.info(read_action, extra={'msg_type': 'ACTION'})
                read_obs = runtime.run_action(read_action)
                logger.info(read_obs, extra={'msg_type': 'OBSERVATION'})

                if (
                    isinstance(read_obs, CmdOutputObservation)
                    and read_obs.exit_code == 0
                ):
                    artifacts[file_path] = read_obs.content.strip()
                elif isinstance(read_obs, CmdOutputObservation):
                    logger.warning(
                        f'Failed to read/encode PoC file {file_path} (exit code {read_obs.exit_code}):\n{read_obs.content}'
                    )
                elif isinstance(read_obs, ErrorObservation):
                    logger.error(
                        f'Error reading/encoding PoC file {file_path}: {read_obs.error_id}'
                    )
                else:
                    logger.error(
                        f'Unexpected observation type when reading/encoding PoC file {file_path}: {type(read_obs)}'
                    )
        else:
            if obs.exit_code != 0:
                logger.warning(
                    f'Failed to list artifacts (exit code {obs.exit_code}):\n{obs.content}'
                )
            else:
                logger.info('No artifacts found in /testcase.')
    elif isinstance(obs, ErrorObservation):
        logger.error(f'Error listing artifacts: {obs.error_id}')
        # Allow continuation, artifacts remains empty
    else:
        assert_and_raise(
            False, f'Unexpected observation type for listing artifacts: {str(obs)}'
        )

    # Function to safely read potentially binary or non-UTF8 files
    def safe_read_file(file_path):
        # First check if the file exists
        check_action = CmdRunAction(
            command=f"""ls -la {file_path} 2>/dev/null || echo "FILE_NOT_FOUND" """
        )
        check_action.set_hard_timeout(10)
        logger.info(check_action, extra={'msg_type': 'ACTION'})
        check_obs = runtime.run_action(check_action)
        logger.info(check_obs, extra={'msg_type': 'OBSERVATION'})

        # If the file doesn't exist, return None
        if isinstance(check_obs, CmdOutputObservation) and (
            check_obs.exit_code != 0 or 'FILE_NOT_FOUND' in check_obs.content
        ):
            logger.warning(f'File does not exist: {file_path}')
            return None

        # File exists, proceed with reading
        # First try to read using cat with base64 encoding to avoid text decoding issues
        action = CmdRunAction(command=f"""cat {file_path} | base64 -w 0""")
        action.set_hard_timeout(30)
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})

        if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
            # Successfully read and base64 encoded
            try:
                import base64

                # Try to decode the base64 content and convert to text if possible
                raw_content = base64.b64decode(obs.content.strip())

                # Try multiple encodings to convert to string
                for encoding in ENCODING_FALLBACKS:
                    try:
                        return raw_content.decode(encoding)
                    except UnicodeDecodeError:
                        continue

                # If all decodings fail, return as hex representation
                logger.warning(
                    f'Could not decode {file_path} as text, returning raw content'
                )
                return raw_content.decode('latin1', errors='replace')
            except Exception as e:
                logger.error(f'Error processing base64 content from {file_path}: {e}')
                return obs.content.strip()  # Return the base64 encoded string
        else:
            logger.warning(
                f'Failed to read {file_path}, command exited with code {obs.exit_code if isinstance(obs, CmdOutputObservation) else "N/A"}'
            )
            return None

    # Attempt to read the patch file generated by the FixerAgent
    patch = safe_read_file('/testcase/model_patch.diff')
    if patch:
        logger.info('Successfully read patch')
    else:
        logger.warning(
            'Failed to read patch or patch not found. Proceeding without patch.'
        )

    # Attempt to read the repository changes diff file
    repo_changes = safe_read_file('/testcase/repo_changes.diff')
    if repo_changes:
        logger.info('Successfully read repository changes diff')
    else:
        logger.info(
            'Failed to read repository changes diff or file not found. Proceeding without repo changes.'
        )

    # Run overall verification for BuilderAgent, ExploiterAgent, and FixerAgent
    # Initialize ExecutionResult with default failure state
    execution_results = ExecutionResult(
        builder={
            'success': False,
            'command': '',
            'exit_code': -1,
            'message': 'Verification not run',
        },
        exploiter={
            'success': False,
            'command': '',
            'exit_code': -1,
            'message': 'Verification not run',
        },
        fixer={
            'success': False,
            'command': '',
            'exit_code': -1,
            'message': 'Verification not run',
        },
    )
    # Extract base commit hash if exists
    action = CmdRunAction(command="""cat /testcase/base_commit_hash""")
    action.set_hard_timeout(30)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})
    if isinstance(obs, CmdOutputObservation) and obs.exit_code == 0:
        base_commit_hash = obs.content.strip()
        logger.info(f'Successfully read base commit hash: {base_commit_hash}')
    else:
        base_commit_hash = instance['base_commit']

    # Verification for BuilderAgent
    builder_success = False
    builder_message = 'Failed verification'

    # Construct the command based on whether repo_changes exists
    build_command = (
        f"""cd {instance['work_dir']} && git reset --hard {base_commit_hash}"""
    )

    # Add the build command
    build_command += """ && secb build"""

    action = CmdRunAction(command=build_command)
    action.set_hard_timeout(900)
    logger.info(action, extra={'msg_type': 'ACTION'})
    obs = runtime.run_action(action)
    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

    if isinstance(obs, CmdOutputObservation):
        if obs.exit_code == 0:
            builder_success = True
            builder_message = 'Build successful at base commit.'
            logger.info('BuilderAgent verification successful.')
        else:
            builder_message = f'Build failed at base commit (exit code {obs.exit_code}):\n{obs.content}'
            logger.warning(f'BuilderAgent verification failed: {builder_message}')
    elif isinstance(obs, ErrorObservation):
        builder_message = f'Error during build verification: {obs.error_id}'
        logger.error(f'BuilderAgent verification error: {builder_message}')
    execution_results.builder = {
        'success': builder_success,
        'command': action.command,
        'exit_code': obs.exit_code,
        'message': builder_message,
    }

    # Verification for ExploiterAgent (only if build succeeded)
    exploiter_success = False
    exploiter_message = 'Skipped due to build failure'
    if builder_success:
        action = CmdRunAction(command='secb repro')
        action.set_hard_timeout(30)  # Allow time for reproduction
        logger.info(action, extra={'msg_type': 'ACTION'})
        obs = runtime.run_action(action)
        logger.info(obs, extra={'msg_type': 'OBSERVATION'})
        if isinstance(obs, CmdOutputObservation):
            output_content = obs.content.strip()
            # Check if any sanitizer pattern exists in the output
            has_sanitizer_report = any(
                pattern in output_content for pattern in SANITIZER_ERROR_PATTERNS
            )

            # TODO: We just check if there is a sanitizer report, but we should check if the exit code is 1
            # if obs.exit_code == 1 and has_sanitizer_report:
            if has_sanitizer_report:
                exploiter_success = True
                exploiter_message = (
                    'Reproduction successful with sanitizer report at base commit.'
                )
                logger.info('ExploiterAgent verification successful.')
            else:
                exploiter_message = f'Reproduction failed at base commit (exit code {obs.exit_code}):\n{obs.content}'
                logger.warning(
                    f'ExploiterAgent verification failed: {exploiter_message}'
                )
        elif isinstance(obs, ErrorObservation):
            exploiter_message = (
                f'Error during reproduction verification: {obs.error_id}'
            )
            logger.error(f'ExploiterAgent verification error: {exploiter_message}')
    else:
        logger.warning(
            'Skipping ExploiterAgent verification due to BuilderAgent failure.'
        )
    execution_results.exploiter = {
        'success': exploiter_success,
        'command': action.command,
        'exit_code': obs.exit_code,
        'message': exploiter_message,
    }

    # Verification for FixerAgent (only if patch commit hash is valid)
    fixer_success = False
    fixer_message = 'Patch has not been found or is invalid'

    if not exploiter_success:
        fixer_message = 'Skipped due to ExploiterAgent verification failure.'
        logger.warning(f'Skipping FixerAgent verification: {fixer_message}')
    else:
        if patch:
            # Run each step separately to capture specific failures
            fixer_steps_results = {
                'patch': {
                    'success': False,
                    'command': '',
                    'exit_code': -1,
                    'output': '',
                },
                'build': {
                    'success': False,
                    'command': '',
                    'exit_code': -1,
                    'output': '',
                },
                'repro': {
                    'success': False,
                    'command': '',
                    'exit_code': -1,
                    'output': '',
                },
            }

            # Step 1: Start with a clean state and apply patches in sequence
            total_commands = []
            patch_cmd = f"""cd {instance['work_dir']} && git clean -fd && git reset --hard {base_commit_hash}"""

            # Then apply the model's vulnerability fix patch
            patch_cmd += """ && secb patch"""

            action = CmdRunAction(command=patch_cmd)
            action.set_hard_timeout(60)
            logger.info(action, extra={'msg_type': 'ACTION'})
            obs = runtime.run_action(action)
            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

            if isinstance(obs, CmdOutputObservation):
                fixer_steps_results['patch']['exit_code'] = obs.exit_code
                fixer_steps_results['patch']['output'] = obs.content.strip()
                fixer_steps_results['patch']['success'] = obs.exit_code == 0
                fixer_steps_results['patch']['command'] = action.command
                total_commands.append(action.command)

                if not fixer_steps_results['patch']['success']:
                    fixer_message = f'Patch application failed (exit code {obs.exit_code}):\n{obs.content}'
                    logger.warning(f'FixerAgent patch step failed: {fixer_message}')
                else:
                    # Step 2: Build with the patch (only if patch succeeded)
                    build_cmd = f"""cd {instance['work_dir']} && secb build"""
                    action = CmdRunAction(command=build_cmd)
                    action.set_hard_timeout(900)
                    logger.info(action, extra={'msg_type': 'ACTION'})
                    obs = runtime.run_action(action)
                    logger.info(obs, extra={'msg_type': 'OBSERVATION'})

                    if isinstance(obs, CmdOutputObservation):
                        fixer_steps_results['build']['exit_code'] = obs.exit_code
                        fixer_steps_results['build']['output'] = obs.content.strip()
                        fixer_steps_results['build']['success'] = obs.exit_code == 0
                        fixer_steps_results['build']['command'] = action.command
                        total_commands.append(action.command)

                        if not fixer_steps_results['build']['success']:
                            fixer_message = f'Build after patch failed (exit code {obs.exit_code}):\n{obs.content}'
                            logger.warning(
                                f'FixerAgent build step failed: {fixer_message}'
                            )
                        else:
                            # Step 3: Run reproduction test with the patch (only if build succeeded)
                            repro_cmd = f"""cd {instance['work_dir']} && secb repro"""
                            action = CmdRunAction(command=repro_cmd)
                            action.set_hard_timeout(30)
                            logger.info(action, extra={'msg_type': 'ACTION'})
                            obs = runtime.run_action(action)
                            logger.info(obs, extra={'msg_type': 'OBSERVATION'})

                            if isinstance(obs, CmdOutputObservation):
                                output_content = obs.content.strip()
                                fixer_steps_results['repro']['exit_code'] = (
                                    obs.exit_code
                                )
                                fixer_steps_results['repro']['output'] = output_content
                                fixer_steps_results['repro']['command'] = action.command
                                total_commands.append(action.command)

                                # Check if any sanitizer pattern exists in the output
                                has_sanitizer_report = any(
                                    pattern in output_content
                                    for pattern in SANITIZER_ERROR_PATTERNS
                                )

                                # For repro, success means no sanitizer report (vulnerability fixed)
                                fixer_steps_results['repro'][
                                    'success'
                                ] = not has_sanitizer_report

                                if fixer_steps_results['repro']['success']:
                                    fixer_success = True
                                    fixer_message = 'Patch fixed the vulnerability - no sanitizer errors detected during reproduction.'
                                    logger.info('FixerAgent verification successful.')
                                else:
                                    fixer_message = 'Patch did not fix the vulnerability - sanitizer errors still detected during reproduction.'
                                    logger.warning(
                                        f'FixerAgent verification failed: {fixer_message}'
                                    )
                            elif isinstance(obs, ErrorObservation):
                                fixer_steps_results['repro']['output'] = (
                                    f'Error: {obs.error_id}'
                                )
                                fixer_message = (
                                    f'Error during reproduction step: {obs.error_id}'
                                )
                                logger.error(
                                    f'FixerAgent repro step error: {fixer_message}'
                                )
                    elif isinstance(obs, ErrorObservation):
                        fixer_steps_results['build']['output'] = (
                            f'Error: {obs.error_id}'
                        )
                        fixer_message = f'Error during build step: {obs.error_id}'
                        logger.error(f'FixerAgent build step error: {fixer_message}')
            elif isinstance(obs, ErrorObservation):
                fixer_steps_results['patch']['output'] = f'Error: {obs.error_id}'
                fixer_message = f'Error during patch application: {obs.error_id}'
                logger.error(f'FixerAgent patch step error: {fixer_message}')

            # Store detailed step results in execution_results
            execution_results.fixer = {
                'success': fixer_success,
                'message': fixer_message,
                'steps': fixer_steps_results,
                'command': ' && '.join(total_commands),
                'exit_code': fixer_steps_results['patch'][
                    'exit_code'
                ],  # Store the first exit code for backward compatibility
            }
        else:
            fixer_message = 'No patch file found to apply.'
            logger.warning(f'FixerAgent verification skipped: {fixer_message}')
            execution_results.fixer = {
                'success': False,
                'command': '',
                'exit_code': -1,
                'message': fixer_message,
            }

    logger.info('-' * 30)
    logger.info('END Runtime Completion Fn')
    logger.info('-' * 30)

    # Return None if essential scripts weren't retrieved
    if build_script is None or secb_script is None:  # Keep artifacts optional for now
        logger.error(
            'Failed to retrieve essential build or secb script content. Returning None.'
        )
        return None

    return InstanceOutput(
        execution=execution_results,  # Add the execution results
        build_sh=build_script,
        secb_sh=secb_script,
        artifacts=artifacts,
        env=env,
        base_commit_hash=base_commit_hash,
        patch=patch,
        repo_changes=repo_changes,  # Add the repository changes
    )


def main(
    llm_config_arg: str,
    condenser_type: str,
    max_iterations: int,
    max_budget_per_task: float,
    dataset_name: str,
    headless: bool = False,
    output_dir: str = './outputs',
    instance_id: Optional[str] = None,
    limit: Optional[str] = None,
    label: Optional[str] = None,
    num_workers: int = 1,
) -> None:
    """Main function to run the Reproducer system.

    Args:
        llm_config_arg: The name of the LLM config to use
        condenser_type: The type of condenser to use
        max_iterations: Maximum number of agent iterations
        max_budget_per_task: Maximum budget per task
        dataset_name: The name of the dataset to use
        headless: Whether to run in headless mode
        output_dir: Directory to save logs and trajectory files
        instance_id: Optional instance ID to process (for debugging)
        limit: Optional limit string on number of instances to process (e.g., ":10", "5:15", "20:")
        label: Optional label to use for the run
        num_workers: Number of workers to use for parallel processing
    """
    # Load the dataset
    dataset = load_dataset(dataset_name)
    df = dataset[label].to_pandas()

    # Filter by instance_id if provided
    if instance_id:
        df = df[df['instance_id'] == instance_id]
        if len(df) == 0:
            logger.error(f'No instances found with ID: {instance_id}')
            return

    # Apply limit if provided
    if limit:
        try:
            start, end = None, None
            if ':' in limit:
                parts = limit.split(':')
                if len(parts) != 2:
                    raise ValueError(
                        "Limit range must have one colon (e.g., ':N', 'M:', 'M:N')"
                    )

                start_str, end_str = parts

                if start_str:
                    start = int(start_str)
                    if start < 0:
                        raise ValueError('Start index cannot be negative')
                if end_str:
                    end = int(end_str)
                    if end < 0:
                        raise ValueError('End index cannot be negative')

                if start is not None and end is not None and start >= end:
                    raise ValueError(
                        'Start index must be less than end index in M:N range'
                    )

            else:
                # Treat as simple limit ":N" if no colon
                end = int(limit)
                if end <= 0:
                    raise ValueError('Simple limit N must be positive')

            original_count = len(df)
            if start is None and end is not None:  # :N
                df = df.iloc[:end]
                logger.info(
                    f'Applying limit: Processing first {len(df)} instances (range : {end}).'
                )
            elif start is not None and end is None:  # M:
                df = df.iloc[start:]
                logger.info(
                    f'Applying limit: Processing instances from index {start} onwards (range {start}:).'
                )
            elif start is not None and end is not None:  # M:N
                df = df.iloc[start:end]
                logger.info(
                    f'Applying limit: Processing instances from index {start} to {end - 1} (range {start}:{end}).'
                )
            else:  # Should not happen if validation is correct, but handle just in case
                logger.warning(
                    f"Could not parse limit '{limit}', processing all instances."
                )

            if len(df) == 0 and original_count > 0:
                logger.warning(
                    f"Limit '{limit}' resulted in zero instances to process from the original {original_count}."
                )

        except ValueError as e:
            logger.error(
                f"Invalid limit format '{limit}': {e}. Processing all instances."
            )

    df = filter_dataset(df, 'instance_id')

    logger.info(f'Processing {len(df)} instance(s)')

    # Select and configure LLM based on argument
    llm_config = get_llm_config_arg(llm_config_arg)
    if llm_config is None:
        # Fallback to default LLM config from OpenHands config.toml
        oh_cfg = load_openhands_config(config_file='config.toml')
        llm_config = oh_cfg.get_llm_config()
        logger.info(
            f"LLM config '{llm_config_arg}' not found in config.toml. Falling back to default [llm] config."
        )
    assert llm_config is not None
    llm_config.log_completions = True

    # Select and configure condenser based on argument
    condenser_config: CondenserConfig

    if condenser_type == 'recent':
        condenser_config = RecentEventsCondenserConfig(
            type='recent', keep_first=1, max_events=MAX_EVENTS_TO_KEEP
        )
    elif condenser_type == 'observation_masking':
        condenser_config = ObservationMaskingCondenserConfig(
            type='observation_masking', attention_window=MAX_EVENTS_TO_KEEP
        )
    elif condenser_type == 'llm_attention':
        condenser_config = LLMAttentionCondenserConfig(
            type='llm_attention',
            llm_config=llm_config,
            max_size=MAX_EVENTS_TO_KEEP,
            keep_first=1,
        )
    elif condenser_type == 'llm':
        condenser_config = LLMSummarizingCondenserConfig(
            type='llm', llm_config=llm_config
        )
    else:
        logger.warning(
            f"Unknown condenser type '{condenser_type}', defaulting to observation_masking."
        )
        condenser_config = ObservationMaskingCondenserConfig(
            type='observation_masking', attention_window=MAX_EVENTS_TO_KEEP
        )

    # Create metadata for the evaluation
    metadata = make_metadata(
        llm_config=llm_config,
        dataset_name=dataset_name,
        agent_class='MultiAgent',
        max_iterations=max_iterations,
        eval_note=f'condenser={condenser_type}',
        eval_output_dir=output_dir,
        details={
            'llm_config_arg': llm_config_arg,
            'condenser_type': condenser_type,
            'max_budget_per_task': max_budget_per_task,
            'headless': headless,
        },
        condenser_config=condenser_config,
    )

    # Output file for results
    output_file = os.path.join(metadata.eval_output_dir, 'output.jsonl')

    # Prepare dataset for processing
    instances = prepare_dataset(df, output_file, eval_n_limit=0)

    # Run evaluation in parallel
    logger.info(f'Running evaluation with {num_workers} workers')
    run_evaluation(
        dataset=instances,
        metadata=metadata,
        output_file=output_file,
        num_workers=num_workers,
        process_instance_func=process_instance,
        timeout_seconds=1800,  # 30 minutes timeout per instance
        max_retries=3,
    )

    logger.info(f'Evaluation complete. Results saved to {output_file}')


if __name__ == '__main__':
    args = parse_args()

    main(
        llm_config_arg=args.llm_config,
        condenser_type=args.condenser,
        max_iterations=args.iterations,
        max_budget_per_task=args.max_budget_per_task,
        dataset_name=args.dataset_name,
        headless=args.headless,
        output_dir=args.output_dir,
        instance_id=args.instance_id,
        limit=args.limit,
        label=args.label,
        num_workers=args.num_workers,
    )
