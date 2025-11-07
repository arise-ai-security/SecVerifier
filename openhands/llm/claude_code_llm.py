"""
Claude Code CLI integration for OpenHands.

This module provides a custom LLM class that uses Claude Code CLI directly
instead of through LiteLLM. This allows leveraging Claude Code's enhanced
coding capabilities including its optimized prompts and tool execution.
"""

import json
import subprocess
import time
import uuid
from pathlib import Path
from typing import Callable

from litellm.types.utils import Choices, ModelResponse, Usage
from litellm.types.utils import Message as LiteLLMMessage

from openhands.core.config import LLMConfig
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics


class ClaudeCodeLLM(LLM):
    """
    Custom LLM that uses Claude Code CLI for completion requests.

    This class maintains a persistent session with Claude Code and handles:
    - Message format translation (OpenHands ↔ Claude Code)
    - Tool execution parsing and logging
    - Conversation history management
    - Thinking/reasoning log capture
    - Cost and usage tracking
    """

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        """
        Initialize Claude Code LLM.

        Args:
            config: LLM configuration (model, api_key, etc.)
            metrics: Metrics tracking instance
            retry_listener: Optional callback for retry events
        """
        # Don't call super().__init__() to avoid LiteLLM initialization
        # Instead, set up the necessary attributes directly
        self.config = config
        self.metrics = (
            metrics if metrics is not None else Metrics(model_name=config.model)
        )
        self.retry_listener = retry_listener
        self.cost_metric_supported = True
        self._tried_model_info = False
        self.model_info = None

        # Claude Code specific attributes
        self.session_id: str | None = None
        self.conversation_log: list[dict] = []
        self.thinking_log_path: Path | None = None
        self.workspace_dir: str | None = None

        # Track total cost and usage
        self.total_cost_usd = 0.0
        self.total_input_tokens = 0
        self.total_output_tokens = 0

        logger.info(f'Claude Code LLM initialized with model: {config.model}')

    def start_session(self, workspace_dir: str, log_dir: str):
        """
        Initialize Claude Code session.

        Args:
            workspace_dir: Working directory for Claude Code
            log_dir: Directory to save thinking and conversation logs
        """
        self.workspace_dir = workspace_dir
        self.session_id = str(uuid.uuid4())

        # Set up logging paths
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        self.thinking_log_path = log_path / 'claude_code_conversation.jsonl'

        logger.info(
            f'Claude Code session started: {self.session_id} in {workspace_dir}'
        )

    def close_session(self):
        """Clean up Claude Code session and save logs."""
        if self.conversation_log and self.thinking_log_path:
            self._save_conversation_log()

        logger.info(
            f'Claude Code session closed. Total cost: ${self.total_cost_usd:.4f}, '
            f'Tokens: {self.total_input_tokens} in / {self.total_output_tokens} out'
        )

        self.session_id = None
        self.conversation_log = []

    @property
    def completion(self) -> Callable:
        """
        Return the completion function.
        This property is required by OpenHands' LLM interface.
        """
        return self._completion

    def _completion(
        self, messages: list[dict], tools: list[dict] | None = None, **kwargs
    ) -> ModelResponse:
        """
        Main completion method called by OpenHands agents.

        Args:
            messages: List of message dicts with 'role' and 'content'
            tools: List of available tools (not used - Claude Code uses its own)
            **kwargs: Additional parameters

        Returns:
            ModelResponse object compatible with LiteLLM format
        """
        if not self.session_id:
            raise RuntimeError(
                'Claude Code session not started. Call start_session() first.'
            )

        start_time = time.time()

        # Convert messages to prompt
        prompt = self._convert_messages_to_prompt(messages)

        logger.debug(
            f'Sending to Claude Code (session {self.session_id}): {len(prompt)} chars'
        )

        # Call Claude Code CLI
        response_data = self._call_claude_code_cli(prompt)

        # Log conversation
        self.conversation_log.append(
            {
                'timestamp': time.time(),
                'prompt_length': len(prompt),
                'response': response_data,
                'messages_count': len(messages),
            }
        )

        # Parse response to ModelResponse format
        model_response = self._parse_response_to_model_response(response_data)

        # Update metrics
        self._update_metrics(response_data, time.time() - start_time)

        # Log completion if enabled
        if self.config.log_completions and self.config.log_completions_folder:
            self._log_completion(messages, model_response)

        return model_response

    def _convert_messages_to_prompt(self, messages: list[dict]) -> str:
        """
        Convert OpenHands messages to Claude Code prompt format.

        Claude Code CLI expects a simple text prompt. We combine all messages
        into a context-aware prompt that includes previous exchanges.
        """
        prompt_parts = []

        for msg in messages:
            # Handle both dict and Pydantic Message objects
            if isinstance(msg, dict):
                role = msg.get('role', '')
                content = msg.get('content', '')
                tool_name = msg.get('name', 'unknown')
            else:
                # Pydantic Message object - access attributes directly
                role = getattr(msg, 'role', '')
                content = getattr(msg, 'content', '')
                tool_name = getattr(msg, 'name', 'unknown')

            if isinstance(content, list):
                # Handle content array (multimodal messages)
                text_parts = []
                for item in content:
                    if isinstance(item, dict):
                        if item.get('type') == 'text':
                            text_parts.append(item.get('text', ''))
                    else:
                        # Pydantic object
                        if getattr(item, 'type', None) == 'text':
                            text_parts.append(getattr(item, 'text', ''))
                content = '\n'.join(text_parts)

            if role == 'system':
                # System messages provide context
                prompt_parts.append(f'<system_context>\n{content}\n</system_context>')
            elif role == 'user':
                prompt_parts.append(f'User request:\n{content}')
            elif role == 'assistant':
                prompt_parts.append(f'Previous response:\n{content}')
            elif role == 'tool':
                # Tool results from previous executions
                prompt_parts.append(f'Result from {tool_name}:\n{content}')

        return '\n\n'.join(prompt_parts)

    def _call_claude_code_cli(self, prompt: str) -> dict:
        """
        Execute Claude Code CLI and parse response.

        Uses --print mode with --output-format=json for programmatic access.
        """
        cmd: list[str] = [
            'claude',
            '-p',  # Print mode (non-interactive)
            '--output-format',
            'json',  # Structured output
            '--session-id',
            str(self.session_id),  # Continue session
            '--dangerously-skip-permissions',  # Auto-approve in container
        ]

        # Add workspace if set
        if self.workspace_dir:
            cmd.extend(['--add-dir', self.workspace_dir])

        # Execute
        try:
            result = subprocess.run(
                cmd,
                input=prompt,
                capture_output=True,
                text=True,
                timeout=300,  # 5 minute timeout
                check=False,
            )

            if result.returncode != 0:
                logger.error(f'Claude Code CLI error: {result.stderr}')
                raise RuntimeError(f'Claude Code CLI failed: {result.stderr}')

            # Parse JSON response
            response_data = json.loads(result.stdout)
            return response_data

        except subprocess.TimeoutExpired:
            logger.error('Claude Code CLI timeout')
            raise RuntimeError('Claude Code CLI timeout after 300s')
        except json.JSONDecodeError as e:
            logger.error(f'Failed to parse Claude Code response: {e}')
            logger.error(f'Raw output: {result.stdout[:500]}')
            raise RuntimeError(f'Invalid JSON from Claude Code: {e}')

    def _parse_response_to_model_response(self, response_data: dict) -> ModelResponse:
        """
        Convert Claude Code response to OpenHands ModelResponse format.

        Returns a LiteLLM ModelResponse object that OpenHands agents expect.
        """
        # Extract result text
        result_text = response_data.get('result', '')

        # Extract usage info
        usage_data = response_data.get('usage', {})
        input_tokens = usage_data.get('input_tokens', 0) + usage_data.get(
            'cache_read_input_tokens', 0
        )
        output_tokens = usage_data.get('output_tokens', 0)

        # Check for permission denials (tool execution failures)
        if response_data.get('permission_denials'):
            denials = response_data['permission_denials']
            logger.warning(f'Claude Code had {len(denials)} permission denials')
            result_text += f'\n\n[Note: {len(denials)} tool executions were blocked by permissions]'

        # Build ModelResponse object
        model_response = ModelResponse(
            id=response_data.get('session_id', str(uuid.uuid4())),
            object='chat.completion',
            created=int(time.time()),
            model=self.config.model,
            choices=[
                Choices(
                    index=0,
                    message=LiteLLMMessage(
                        role='assistant',
                        content=result_text,
                    ),
                    finish_reason='stop',
                )
            ],
            usage=Usage(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=input_tokens + output_tokens,
            ),
        )

        return model_response

    def _update_metrics(self, response_data: dict, duration: float):
        """Update cost and usage metrics."""
        cost = response_data.get('total_cost_usd', 0.0)
        usage = response_data.get('usage', {})

        input_tokens = usage.get('input_tokens', 0) + usage.get(
            'cache_read_input_tokens', 0
        )
        output_tokens = usage.get('output_tokens', 0)

        self.total_cost_usd += cost
        self.total_input_tokens += input_tokens
        self.total_output_tokens += output_tokens

        # Update metrics object
        if self.metrics:
            self.metrics.add_cost(cost)
            # Note: add_token_usage requires more params, tracking internally for now

        logger.debug(
            f'Claude Code call: ${cost:.4f}, {input_tokens} in / {output_tokens} out tokens, '
            f'{duration:.2f}s'
        )

    def _log_completion(self, messages: list[dict], response: ModelResponse):
        """Log completion to file if enabled."""
        log_file = (
            Path(self.config.log_completions_folder) / f'default-{time.time()}.json'
        )

        # Convert Pydantic Message objects to dicts for JSON serialization
        serializable_messages = []
        for msg in messages:
            if isinstance(msg, dict):
                serializable_messages.append(msg)
            else:
                # Pydantic object - convert to dict
                if hasattr(msg, 'model_dump'):
                    serializable_messages.append(msg.model_dump())
                elif hasattr(msg, 'dict'):
                    serializable_messages.append(msg.dict())
                else:
                    # Fallback: manually extract attributes
                    serializable_messages.append(
                        {
                            'role': getattr(msg, 'role', ''),
                            'content': getattr(msg, 'content', ''),
                        }
                    )

        # Convert ModelResponse to dict for JSON serialization
        if hasattr(response, 'model_dump'):
            serializable_response = response.model_dump()
        elif hasattr(response, 'dict'):
            serializable_response = response.dict()
        else:
            serializable_response = dict(response)

        log_data = {
            'timestamp': time.time(),
            'messages': serializable_messages,
            'response': serializable_response,
            'session_id': self.session_id,
        }

        with open(log_file, 'w') as f:
            json.dump(log_data, f, indent=2)

    def _save_conversation_log(self):
        """Save conversation history to JSONL file."""
        if not self.thinking_log_path:
            return

        with open(self.thinking_log_path, 'w') as f:
            for entry in self.conversation_log:
                f.write(json.dumps(entry) + '\n')

        logger.info(f'Conversation log saved to {self.thinking_log_path}')

    # Implement required LLM interface methods

    def vision_is_active(self) -> bool:
        """Claude Code supports vision through Claude models."""
        return (
            'sonnet' in self.config.model.lower() or 'opus' in self.config.model.lower()
        )

    def is_caching_prompt_active(self) -> bool:
        """Claude Code uses prompt caching automatically."""
        return True

    def is_function_calling_active(self) -> bool:
        """Claude Code has its own tool system."""
        return True

    def format_messages_for_llm(self, messages) -> list[dict]:
        """
        Format messages for LLM (passthrough for Claude Code).

        This is called by OpenHands before sending to completion().
        We just return as-is since we handle formatting in _convert_messages_to_prompt().
        """
        if isinstance(messages, list):
            return messages
        return [messages] if messages else []
