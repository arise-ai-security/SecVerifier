"""Claude Code SDK integration for OpenHands.

This module provides a wrapper around the Claude Code SDK (claude-agent-sdk-python)
to integrate Claude Code as a backbone LLM for OpenHands agents.

Key Features:
- Converts OpenHands message format to Claude Code SDK event format
- Handles tool calling through Claude Code's native tool support
- Returns litellm-compatible ModelResponse objects
- Supports all OpenHands agent capabilities (BuilderAgent, ExploiterAgent, FixerAgent)
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncIterator, Callable, Sequence
from uuid import uuid4

from litellm.types.utils import Choices, ModelResponse, Usage
from litellm.types.utils import Message as LiteLLMMessage

from openhands.core.config import LLMConfig
from openhands.core.exceptions import LLMNoResponseError
from openhands.core.logger import openhands_logger as logger
from openhands.llm.llm import LLM
from openhands.llm.metrics import Metrics

try:
    from claude_agent_sdk import query as claude_code_query
    from claude_agent_sdk._errors import CLIConnectionError, CLINotFoundError
    from claude_agent_sdk.types import (
        AssistantMessage,
        ClaudeAgentOptions,
        ResultMessage,
        TextBlock,
        ThinkingBlock,
        ToolResultBlock,
        ToolUseBlock,
    )
    from claude_agent_sdk.types import (
        Message as ClaudeMessage,
    )

    CLAUDE_SDK_AVAILABLE = True
except ImportError:
    CLAUDE_SDK_AVAILABLE = False
    logger.warning(
        'claude-agent-sdk not installed. Install with: pip install claude-agent-sdk'
    )


def _stringify_content(content: Any) -> str:
    """Convert any content type to string for Claude Code SDK.

    Args:
        content: Content to stringify (str, dict, list, etc.)

    Returns:
        String representation of the content
    """
    if content is None:
        return ''
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                text = item.get('text')
                if text is not None:
                    parts.append(str(text))
                elif 'type' in item and 'content' in item:
                    parts.append(json.dumps(item))
            else:
                parts.append(str(item))
        return '\n'.join(parts)
    if isinstance(content, dict):
        try:
            return json.dumps(content)
        except Exception:
            return str(content)
    return str(content)


def _safe_json_dumps(data: Any) -> str:
    """Safely dump data to JSON string.

    Args:
        data: Data to serialize

    Returns:
        JSON string representation
    """
    try:
        return json.dumps(data)
    except Exception:
        return json.dumps({'value': _stringify_content(data)})


def _normalize_tool_arguments(arguments: Any) -> dict[str, Any]:
    """Normalize tool arguments to dict format.

    Args:
        arguments: Tool arguments (dict, str, or other)

    Returns:
        Dictionary of arguments
    """
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            return json.loads(arguments)
        except json.JSONDecodeError:
            return {'arguments': arguments}
    return {'arguments': arguments}


def _convert_tool_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert OpenHands tool message to Claude SDK format.

    Args:
        message: Tool message from OpenHands

    Returns:
        Claude SDK format tool_result event
    """
    tool_use_id = message.get('tool_call_id') or message.get('tool_use_id')
    block: dict[str, Any] = {
        'type': 'tool_result',
        'tool_use_id': tool_use_id,
    }
    if 'is_error' in message:
        block['is_error'] = message['is_error']

    content = message.get('content')
    if isinstance(content, list):
        block['content'] = []
        for item in content:
            if isinstance(item, dict):
                block['content'].append(item)
            else:
                block['content'].append({'type': 'text', 'text': str(item)})
    elif content is not None:
        block['content'] = content
    elif message.get('text'):
        block['content'] = message['text']

    return {
        'type': 'user',
        'message': {
            'role': 'user',
            'content': [block],
        },
    }


def _extract_text_blocks(content: Any) -> list[dict[str, Any]]:
    """Extract text blocks from message content.

    Args:
        content: Message content (str, list, dict)

    Returns:
        List of text block dictionaries
    """
    blocks: list[dict[str, Any]] = []
    if content is None:
        return blocks
    if isinstance(content, str):
        if content:
            blocks.append({'type': 'text', 'text': content})
        return blocks
    if isinstance(content, list):
        for item in content:
            if isinstance(item, dict):
                if item.get('type') == 'text' and 'text' in item:
                    blocks.append({'type': 'text', 'text': item['text']})
                elif item.get('type') == 'input_text' and 'text' in item:
                    blocks.append({'type': 'text', 'text': item['text']})
    elif isinstance(content, dict) and 'text' in content:
        blocks.append({'type': 'text', 'text': str(content['text'])})
    else:
        blocks.append({'type': 'text', 'text': _stringify_content(content)})
    return blocks


def _convert_user_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert OpenHands user message to Claude SDK format.

    Args:
        message: User message from OpenHands

    Returns:
        Claude SDK format user event or None if empty
    """
    blocks = _extract_text_blocks(message.get('content'))
    if not blocks:
        text = _stringify_content(message.get('content'))
        if not text:
            return None
        return {'type': 'user', 'message': {'role': 'user', 'content': text}}
    return {'type': 'user', 'message': {'role': 'user', 'content': blocks}}


def _convert_assistant_message(message: dict[str, Any]) -> dict[str, Any] | None:
    """Convert OpenHands assistant message to Claude SDK format.

    Args:
        message: Assistant message from OpenHands

    Returns:
        Claude SDK format assistant event or None if empty
    """
    content_blocks: list[dict[str, Any]] = []

    # Handle tool_calls
    tool_calls = message.get('tool_calls') or []
    for tool_call in tool_calls:
        function_data = tool_call.get('function', {})
        call_id = tool_call.get('id') or function_data.get('id') or str(uuid4())
        content_blocks.append(
            {
                'type': 'tool_use',
                'id': call_id,
                'name': function_data.get('name', ''),
                'input': _normalize_tool_arguments(function_data.get('arguments', {})),
            }
        )

    # Handle function_call (legacy format)
    if message.get('function_call'):
        function_call = message['function_call']
        call_id = function_call.get('id') or str(uuid4())
        content_blocks.append(
            {
                'type': 'tool_use',
                'id': call_id,
                'name': function_call.get('name', ''),
                'input': _normalize_tool_arguments(function_call.get('arguments', {})),
            }
        )

    # Add text content
    content_blocks.extend(_extract_text_blocks(message.get('content')))

    if not content_blocks:
        return None

    return {
        'type': 'assistant',
        'message': {
            'role': 'assistant',
            'content': content_blocks,
        },
    }


def _prepare_prompt_stream(
    messages: Sequence[dict[str, Any]],
) -> tuple[str | None, list[dict[str, Any]]]:
    """Prepare OpenHands messages for Claude Code SDK.

    Converts OpenHands message format to Claude SDK event stream format.
    Extracts system prompts and converts messages to events.

    Args:
        messages: List of OpenHands messages

    Returns:
        Tuple of (system_prompt, events_list)
    """
    system_parts: list[str] = []
    events: list[dict[str, Any]] = []

    for message in messages:
        role = message.get('role')
        if role == 'system':
            system_parts.append(_stringify_content(message.get('content')))
            continue
        if role == 'tool':
            tool_event = _convert_tool_message(message)
            events.append(tool_event)
            continue
        if role == 'assistant':
            assistant_event = _convert_assistant_message(message)
            if assistant_event:
                events.append(assistant_event)
            continue
        # Default to user message
        user_event = _convert_user_message(message)
        if user_event:
            events.append(user_event)

    system_prompt = '\n\n'.join([part for part in system_parts if part.strip()]) or None
    return system_prompt, events


async def _run_query_async(
    events: Sequence[dict[str, Any]],
    options: ClaudeAgentOptions,
) -> list[ClaudeMessage]:
    """Run Claude Code query asynchronously.

    Args:
        events: List of Claude SDK events
        options: Claude agent options

    Returns:
        List of Claude messages from the SDK
    """

    async def _stream() -> AsyncIterator[dict[str, Any]]:
        for event in events:
            yield event

    collected: list[ClaudeMessage] = []
    async for item in claude_code_query(prompt=_stream(), options=options):
        collected.append(item)
    return collected


def _run_query_sync(
    events: Sequence[dict[str, Any]], options: ClaudeAgentOptions
) -> list[ClaudeMessage]:
    """Run Claude Code query synchronously.

    Args:
        events: List of Claude SDK events
        options: Claude agent options

    Returns:
        List of Claude messages from the SDK
    """

    async def _runner() -> list[ClaudeMessage]:
        return await _run_query_async(events, options)

    try:
        return asyncio.run(_runner())
    except RuntimeError:
        # Create new event loop if current one is closed
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(_runner())
        finally:
            loop.close()


def _to_model_response(
    assistant_message: AssistantMessage,
    result_message: ResultMessage | None,
    model: str,
) -> ModelResponse:
    """Convert Claude SDK response to litellm ModelResponse.

    Args:
        assistant_message: Assistant message from Claude SDK
        result_message: Result message with usage/cost info
        model: Model name

    Returns:
        litellm-compatible ModelResponse
    """
    text_parts: list[str] = []
    tool_calls_for_message: list[dict[str, Any]] = []

    for block in assistant_message.content:
        if isinstance(block, TextBlock):
            text_parts.append(block.text)
        elif isinstance(block, ToolUseBlock):
            tool_calls_for_message.append(
                {
                    'id': block.id or '',
                    'type': 'function',
                    'function': {
                        'name': block.name,
                        'arguments': _safe_json_dumps(block.input),
                    },
                }
            )
        elif isinstance(block, ToolResultBlock):
            # Surface tool results as additional text for visibility
            if block.content:
                text_parts.append(_stringify_content(block.content))
        elif isinstance(block, ThinkingBlock):
            # Log thinking blocks but don't include in output
            logger.debug(
                f'Claude Code thinking block received ({len(block.thinking)} chars)'
            )

    content = ''.join(text_parts).strip()
    message_payload = LiteLLMMessage(
        content=content if content else None,
        role='assistant',
        tool_calls=tool_calls_for_message or None,
    )
    finish_reason = 'tool_calls' if tool_calls_for_message else 'stop'
    choice = Choices(message=message_payload, finish_reason=finish_reason)

    usage_kwargs: dict[str, Any] = {}
    if result_message is not None:
        usage_data = result_message.usage or {}
        prompt_tokens = (
            usage_data.get('input_tokens') or usage_data.get('prompt_tokens') or 0
        )
        completion_tokens = (
            usage_data.get('output_tokens') or usage_data.get('completion_tokens') or 0
        )
        total_tokens = usage_data.get('total_tokens') or (
            prompt_tokens + completion_tokens
        )
        usage_kwargs.update(
            {
                'prompt_tokens': prompt_tokens,
                'completion_tokens': completion_tokens,
                'total_tokens': total_tokens,
            }
        )
        # Add cache tokens if available
        if usage_data.get('cache_creation_input_tokens') is not None:
            usage_kwargs['cache_creation_input_tokens'] = usage_data[
                'cache_creation_input_tokens'
            ]
        if usage_data.get('cache_read_input_tokens') is not None:
            usage_kwargs['cache_read_input_tokens'] = usage_data[
                'cache_read_input_tokens'
            ]

    usage = Usage(**usage_kwargs)
    response = ModelResponse(model=model, choices=[choice], usage=usage)

    # Add cost information if available
    if result_message and result_message.total_cost_usd is not None:
        cost = result_message.total_cost_usd
        response._hidden_params = {  # type: ignore[attr-defined]
            'additional_headers': {
                'llm_provider-x-litellm-response-cost': str(cost),
            }
        }
        logger.debug(f'Claude Code cost: ${cost:.4f}')

    return response


class ClaudeCodeLLM(LLM):
    """Claude Code SDK-based LLM implementation for OpenHands.

    This class integrates the Claude Code SDK (claude-agent-sdk-python) as a
    backend LLM provider for OpenHands agents. It wraps the base LLM class and
    overrides the completion method to use Claude Code instead of litellm.

    Configuration:
        Set custom_llm_provider = "claude_code" in LLMConfig to use this provider.

    Environment Variables:
        - CLAUDE_CODE_CLI_PATH: Path to Claude Code CLI binary (optional)
        - CLAUDE_CODE_CWD: Working directory for Claude Code (optional)
        - CLAUDE_CODE_PERMISSION_MODE: Permission mode (default: 'bypassPermissions')
        - ANTHROPIC_API_KEY: API key for Claude (required)

    Example:
        ```toml
        [llm.claude_agent]
        model = "claude-sonnet-4-5-20250929"
        custom_llm_provider = "claude_code"
        native_tool_calling = true
        ```
    """

    def __init__(
        self,
        config: LLMConfig,
        metrics: Metrics | None = None,
        retry_listener: Callable[[int, int], None] | None = None,
    ) -> None:
        """Initialize Claude Code LLM.

        Args:
            config: LLM configuration
            metrics: Metrics collector
            retry_listener: Callback for retry events
        """
        if not CLAUDE_SDK_AVAILABLE:
            raise RuntimeError(
                'claude-agent-sdk is required for ClaudeCodeLLM. '
                'Install with: pip install claude-agent-sdk'
            )

        # Initialize base LLM (but we won't use its completion method)
        # We need this for metrics, retry logic, and other shared functionality
        super().__init__(config, metrics, retry_listener)

        # Claude Code specific configuration
        self._cli_path = os.getenv('CLAUDE_CODE_CLI_PATH')
        self._cwd = os.getenv('CLAUDE_CODE_CWD', os.getcwd())
        self._permission_mode = os.getenv(
            'CLAUDE_CODE_PERMISSION_MODE', 'bypassPermissions'
        )

        # Force function calling to be active for Claude Code
        if self.config.native_tool_calling is None:
            self._function_calling_active = True
        else:
            self._function_calling_active = self.config.native_tool_calling

        # Session state
        self.workspace_dir: str | None = None
        self.thinking_log_path: str | None = None
        self._session_started = False

        logger.info('Initialized ClaudeCodeLLM')
        logger.info(f'  Model: {self.config.model}')
        logger.info(f'  Permission mode: {self._permission_mode}')
        logger.info(f'  CWD: {self._cwd}')
        logger.info(f'  Function calling: {self._function_calling_active}')

    def start_session(self, workspace_dir: str, log_dir: str) -> None:
        """Start a Claude Code session.

        This method is called by OpenHands to initialize session-specific settings.
        It sets the workspace directory and log directory for the session.

        Args:
            workspace_dir: Directory containing the workspace (e.g., /workspace)
            log_dir: Directory for storing logs and trajectories
        """
        self.workspace_dir = workspace_dir
        self.thinking_log_path = log_dir
        self._session_started = True

        # Update CWD if not explicitly set via environment variable
        if not os.getenv('CLAUDE_CODE_CWD'):
            self._cwd = workspace_dir

        logger.info('Claude Code session started:')
        logger.info(f'  Workspace: {self.workspace_dir}')
        logger.info(f'  Log directory: {self.thinking_log_path}')
        logger.info(f'  CWD: {self._cwd}')

    def end_session(self) -> None:
        """End a Claude Code session.

        This method is called when the session is complete to clean up resources.
        """
        if self._session_started:
            logger.info('Claude Code session ended')
            self._session_started = False

    def is_function_calling_active(self) -> bool:
        """Check if function calling is active.

        Returns:
            True if function calling is enabled
        """
        return self._function_calling_active

    def _claude_code_complete(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]] | None = None,
        stop: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete using Claude Code SDK.

        Args:
            messages: List of OpenHands messages
            tools: List of available tools
            stop: Stop sequences (currently not used)
            **kwargs: Additional arguments

        Returns:
            litellm-compatible ModelResponse

        Raises:
            LLMNoResponseError: If no response from Claude Code
            RuntimeError: If Claude Code CLI error occurs
        """
        # Prepare messages and system prompt
        system_prompt, events = _prepare_prompt_stream(messages)

        if not events:
            raise LLMNoResponseError('No conversational content to send to Claude Code')

        # Extract allowed tools from the tools parameter
        allowed_tools: list[str] = []
        if tools:
            for tool in tools:
                function_data = tool.get('function')
                if isinstance(function_data, dict) and function_data.get('name'):
                    allowed_tools.append(function_data['name'])

        if allowed_tools:
            allowed_tools = sorted(set(allowed_tools))
            logger.debug(f'Claude Code allowed tools: {allowed_tools}')

        # Configure Claude Agent options
        options = ClaudeAgentOptions()
        options.allowed_tools = allowed_tools
        options.system_prompt = system_prompt
        options.permission_mode = self._permission_mode
        options.model = self.config.model if self.config.model else None
        options.cwd = self._cwd

        if self._cli_path:
            options.cli_path = self._cli_path

        # Run Claude Code query
        try:
            responses = _run_query_sync(events, options)
        except (CLIConnectionError, CLINotFoundError) as exc:
            raise RuntimeError(f'Claude Code CLI error: {exc}') from exc

        # Extract assistant and result messages
        assistant_message: AssistantMessage | None = None
        result_message: ResultMessage | None = None
        for message in responses:
            if isinstance(message, AssistantMessage):
                assistant_message = message
            elif isinstance(message, ResultMessage):
                result_message = message

        if assistant_message is None:
            raise LLMNoResponseError('Claude Code did not return an assistant message')

        # Convert to ModelResponse
        return _to_model_response(assistant_message, result_message, self.config.model)

    def completion(
        self,
        messages: Any = None,
        **kwargs: Any,
    ) -> ModelResponse:
        """Complete a conversation using Claude Code SDK.

        This overrides the base LLM completion method to use Claude Code instead
        of litellm. It maintains compatibility with the OpenHands agent interface.

        Args:
            messages: List of messages or first positional argument
            **kwargs: Additional completion parameters (tools, stop, etc.)

        Returns:
            litellm-compatible ModelResponse
        """
        # Handle messages parameter
        if messages is None:
            messages = kwargs.get('messages', [])
        if not isinstance(messages, list):
            raise ValueError(
                'Claude Code completion expects messages to be a list of dicts'
            )

        # Extract tools and stop parameters
        tools = kwargs.get('tools')
        stop = kwargs.get('stop')

        # Call Claude Code completion
        return self._claude_code_complete(messages, tools, stop, **kwargs)
