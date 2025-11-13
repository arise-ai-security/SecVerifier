"""Builder for creating OpenHands SDK agents with appropriate configuration."""

from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader
from openhands.sdk import Agent, LLM, Tool
from openhands.sdk.context.condenser import LLMSummarizingCondenser
from openhands.tools.file_editor import FileEditorTool
from openhands.tools.task_tracker import TaskTrackerTool
from openhands.tools.terminal import TerminalTool

from sec_bench_sdk.domain.value_objects import AgentType, PhaseConfig


class AgentBuilder:
    """Builds OpenHands SDK agents with phase-specific configuration."""

    def __init__(self, prompt_dir: Path, condenser_type: str = 'recent'):
        """Initialize the agent builder.

        Args:
            prompt_dir: Directory containing Jinja2 prompt templates
            condenser_type: Type of condenser to use ('recent', 'llm', 'none')
        """
        self.prompt_dir = prompt_dir
        self.condenser_type = condenser_type
        self.jinja_env = Environment(
            loader=FileSystemLoader(prompt_dir),
            autoescape=True,
        )

    def build_agent(
        self,
        llm: LLM,
        agent_type: AgentType,
        phase_config: PhaseConfig,
        instance_metadata: Optional[dict] = None,
    ) -> Agent:
        """Build an agent for a specific phase.

        Args:
            llm: LLM instance to use
            agent_type: Type of agent to build
            phase_config: Phase configuration
            instance_metadata: Metadata about the instance being processed

        Returns:
            Configured Agent instance
        """
        # Load and render the system prompt
        system_prompt = self._load_system_prompt(agent_type)

        # Get standard tools for code execution
        tools = self._get_tools_for_agent(agent_type)

        # Create condenser if requested
        condenser = self._create_condenser(llm) if self.condenser_type != 'none' else None

        # Create the agent
        agent = Agent(
            llm=llm,
            tools=tools,
            condenser=condenser,
            # Note: SDK doesn't directly support system_prompt in constructor
            # We'll need to send it as the first message or use a different approach
        )

        return agent

    def _create_condenser(self, llm: LLM):
        """Create a condenser based on the configured type.

        Args:
            llm: LLM instance to use for condenser

        Returns:
            Condenser instance or None
        """
        if self.condenser_type == 'llm':
            # Use LLM-based summarization condenser
            return LLMSummarizingCondenser(
                llm=llm.model_copy(update={"usage_id": "condenser"}),
                max_size=10,  # Trigger when history exceeds 10 events
                keep_first=2,  # Always preserve first 2 events (system prompts)
            )
        elif self.condenser_type == 'recent':
            # Use recent events condenser (default, lightweight)
            # The SDK uses this by default, so we can return None or configure it
            # For now, return None to use SDK default
            return None
        else:
            return None

    def get_system_prompt(self, agent_type: AgentType) -> str:
        """Get the system prompt for an agent type (public method).

        Args:
            agent_type: Type of agent

        Returns:
            System prompt string
        """
        return self._load_system_prompt(agent_type)

    def _load_system_prompt(self, agent_type: AgentType) -> str:
        """Load the system prompt for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            System prompt string
        """
        prompt_files = {
            AgentType.BUILDER: "builder/system_prompt.j2",
            AgentType.EXPLOITER: "exploiter/system_prompt.j2",
            AgentType.FIXER: "fixer/system_prompt.j2",
        }

        template_name = prompt_files.get(agent_type)
        if not template_name:
            return f"You are a {agent_type.value} specialized in security vulnerability analysis."

        try:
            template = self.jinja_env.get_template(template_name)
            return template.render()
        except Exception as e:
            # Fallback to basic prompt if template not found
            return f"You are a {agent_type.value} specialized in security vulnerability analysis."

    def _get_tools_for_agent(self, agent_type: AgentType) -> list[Tool]:
        """Get tools appropriate for an agent type.

        Args:
            agent_type: Type of agent

        Returns:
            List of tools
        """
        # All agents get standard code execution tools
        base_tools = [
            Tool(name=TerminalTool.name),
            Tool(name=FileEditorTool.name),
            Tool(name=TaskTrackerTool.name),
        ]

        # Could add agent-specific tools here
        return base_tools

    def get_task_instruction(
        self,
        agent_type: AgentType,
        instance_metadata: dict,
    ) -> str:
        """Get the task instruction for an agent.

        This method renders the appropriate Jinja2 template with context from
        the instance metadata and previous phases.

        Args:
            agent_type: Type of agent
            instance_metadata: Metadata about the instance (includes previous_output)

        Returns:
            Task instruction string
        """
        instruction_files = {
            AgentType.BUILDER: "instructions/builder_agent_instruction.j2",
            AgentType.EXPLOITER: "instructions/exploiter_agent_instruction.j2",
            AgentType.FIXER: "instructions/fixer_agent_instruction.j2",
        }

        template_name = instruction_files.get(agent_type)
        if not template_name:
            return "Complete your assigned task."

        try:
            # Prepare template context
            template_context = {
                'instance_id': instance_metadata.get('instance_id', 'unknown'),
                'work_dir': '/src/repo',
                'base_commit': instance_metadata.get('base_commit', 'unknown'),
                'bug_description': instance_metadata.get('problem_statement', 'No description provided'),
                'candidate_fixes': self._format_candidate_fixes(instance_metadata),
            }

            # Add previous phase outputs for context
            previous_output = instance_metadata.get('previous_output')
            if previous_output:
                template_context['previous_phase'] = {
                    'type': previous_output.phase_type.value if hasattr(previous_output, 'phase_type') else 'unknown',
                    'success': previous_output.success if hasattr(previous_output, 'success') else False,
                    'artifacts': previous_output.artifacts if hasattr(previous_output, 'artifacts') else {},
                    'outputs': previous_output.outputs if hasattr(previous_output, 'outputs') else {},
                }

            template = self.jinja_env.get_template(template_name)
            return template.render(**template_context)
        except Exception as e:
            # Fallback instruction
            print(f"Warning: Failed to render template {template_name}: {e}")
            return f"Complete the {agent_type.value} task for instance {instance_metadata.get('instance_id', 'unknown')}."

    def _format_candidate_fixes(self, instance_metadata: dict) -> str:
        """Format candidate fix commits for the fixer agent.

        Args:
            instance_metadata: Instance metadata

        Returns:
            Formatted string of candidate fixes
        """
        # Extract candidate fixes from metadata
        fixes = instance_metadata.get('candidate_fixes', [])
        if not fixes:
            fixes = instance_metadata.get('fixing_commit', [])

        if not fixes:
            return "No candidate fix commits provided."

        if isinstance(fixes, str):
            return fixes

        if isinstance(fixes, list):
            return "\n".join(f"- {fix}" for fix in fixes)

        return str(fixes)
