"""
chat/agent.py

Defines the ChatAgent class, responsible for managing the conversation loop,
OpenAI LLM integration, MCP tool-calling, disposable queries, and token tracking.

Key features:
    - OpenAI function/tool calling support.
    - Disposable (non-persistent) query mode with "-d" or "--disposable" prefix.
    - Per-turn token window logging (cost control).
    - Tool call logging for transparency.
    - Robust OpenAI tool/response protocol compliance.

See README.md for architecture overview and example usage.
"""

import tiktoken
import json
import logging
import os
import time
from typing import Dict, List, Any, Optional, Union, TypeVar, Callable, Tuple
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from functools import wraps
import asyncio
from tenacity import retry, stop_after_attempt, wait_exponential
from datetime import datetime
import uuid

from openai import OpenAI
from openai.types.chat import ChatCompletionMessage
from openai import APIError, RateLimitError
from .mcp_client import AlpacaMCPClient

# Configuration
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TOKEN_ENCODING = "cl100k_base"  # Fallback encoding for unknown models
LOG_FILE = "chat_agent.log"
TOKEN_LOG_FILE = "token_usage.log"
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0  # seconds
MAX_TOKENS = 4096  # Maximum tokens per request

# Type variables for generic functions
T = TypeVar('T')

# Ensure log directory exists and create log files
log_path = Path(LOG_FILE)
token_log_path = Path(TOKEN_LOG_FILE)
try:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not log_path.exists():
        log_path.touch()
    if not token_log_path.exists():
        token_log_path.touch()
except (PermissionError, OSError) as e:
    print(f"Warning: Could not create log files: {e}")
    print("Logging will be disabled.")
    LOG_FILE = None
    TOKEN_LOG_FILE = None

# Configure logging to file
if LOG_FILE:
    file_handler = logging.FileHandler(LOG_FILE)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
else:
    file_handler = logging.NullHandler()

# Configure console handler for errors only
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.ERROR)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
logger.addHandler(file_handler)
logger.addHandler(console_handler)

def log_token_usage(prompt_tokens: int, completion_tokens: int, total_tokens: int, model: str) -> None:
    """Log token usage to the token log file."""
    if not TOKEN_LOG_FILE:
        return
        
    timestamp = datetime.now().isoformat()
    log_entry = {
        "timestamp": timestamp,
        "model": model,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "note": "These are the actual tokens counted by OpenAI's API"
    }
    
    with open(TOKEN_LOG_FILE, 'a') as f:
        f.write(json.dumps(log_entry) + '\n')

class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"

@dataclass
class Message:
    role: MessageRole
    content: Any
    tool_calls: Optional[List[Dict]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict:
        """Convert Message to dictionary format for API."""
        message = {"role": self.role.value, "content": self.content}
        if self.tool_calls:
            message["tool_calls"] = self.tool_calls
        if self.tool_call_id:
            message["tool_call_id"] = self.tool_call_id
        return message

SYSTEM_PROMPT = (
    "You are an AI trading assistant that helps users interact with their Alpaca trading account. "
    "You have access to various trading tools through the MCP server. "
    "When users ask about stock prices, account information, or want to place trades, "
    "you MUST use the appropriate tools to get real-time data or execute trades. "
    "Do not make up information or prices - always use the tools to get accurate data. "
    "For example, if asked about a stock price, use the get_quote tool. "
    "If asked to place a trade, use the create_order tool. "
    "If asked about account information, use the get_account tool. "
    "Only respond with general information if no specific tools are needed."
)

class ChatAgentError(Exception):
    """Base exception for ChatAgent errors."""
    pass

class ToolCallError(ChatAgentError):
    """Raised when a tool call fails."""
    pass

class TokenCountError(ChatAgentError):
    """Raised when token counting fails."""
    pass

class RateLimitError(ChatAgentError):
    """Raised when rate limit is exceeded."""
    pass

def rate_limit(func: Callable[..., T]) -> Callable[..., T]:
    """Decorator to implement rate limiting."""
    last_call_time = 0.0
    
    @wraps(func)
    async def wrapper(*args, **kwargs) -> T:
        nonlocal last_call_time
        current_time = time.time()
        time_since_last_call = current_time - last_call_time
        
        if time_since_last_call < RATE_LIMIT_DELAY:
            await asyncio.sleep(RATE_LIMIT_DELAY - time_since_last_call)
        
        result = await func(*args, **kwargs)
        last_call_time = time.time()
        return result
    
    return wrapper

@retry(
    stop=stop_after_attempt(MAX_RETRIES),
    wait=wait_exponential(multiplier=1, min=4, max=10),
    retry_error_callback=lambda retry_state: retry_state.outcome.result()
)
async def retry_on_error(func: Callable[..., T], *args, **kwargs) -> T:
    """Retry a function on failure with exponential backoff."""
    try:
        return await func(*args, **kwargs)
    except (APIError, RateLimitError) as e:
        logger.warning(f"API call failed, retrying: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        raise

def disposable_handler(chat_method):
    async def wrapper(self, user_msg: str, *args, **kwargs):
        disposable = False
        if user_msg.startswith("-d "):
            disposable = True
            user_msg = user_msg[3:].strip()
        elif user_msg.startswith("--disposable "):
            disposable = True
            user_msg = user_msg[len("--disposable "):].strip()
        return await chat_method(self, user_msg, *args, disposable=disposable, **kwargs)
    return wrapper

class ChatAgent:
    """
    Conversational agent integrating OpenAI LLM with Alpaca MCP tool calls.

    Attributes:
        mcp (AlpacaMCPClient): Async client to the running MCP server.
        model (str): OpenAI model name (e.g. "gpt-4o").
        history (List[Dict]): Persisted conversation history (user/assistant/tool messages).
        last_token_usage (Dict): Store the last token usage
        tool_defs (List[Dict]): Cache for tool definitions

    Methods:
        chat(user_msg: str, *, disposable=False) -> str:
            Process user input and return LLM response, handling tools as needed.

        _openai_chat(...):
            Internal utility to call OpenAI API with the constructed message history.

        _print_token_count(messages):
            Print token usage from the last LLM response.
    """
    
    def __init__(self, mcp: AlpacaMCPClient, model: str = DEFAULT_MODEL):
        """Initialize the chat agent with an MCP client."""
        self.mcp = mcp
        self.model = model
        self.openai = OpenAI()
        # Initialize with system message only
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]
        self.last_token_usage = None  # Store the last token usage
        self.tool_defs = None  # Cache for tool definitions
        logger.info(f"Initialized ChatAgent with model: {model}")

    def erase_history(self) -> None:
        """Erase all conversation history except the system message."""
        self.history = [self.history[0]]  # Keep only system message
        logger.info("Conversation history erased")
        print("[HISTORY] All conversation history has been erased.")

    def _update_history(self, message: Dict[str, Any], persist: bool = True) -> List[Dict[str, Any]]:
        """
        Update the conversation history with a new message.
        
        Args:
            message: The message to add
            persist: Whether to persist the message in the main history
            
        Returns:
            The updated history list
        """
        if persist:
            self.history.append(message)
        return self.history.copy()  # Return a copy of the full history

    def _add_message(self, role: MessageRole, content: Any, tool_calls: Optional[List[Dict]] = None, tool_call_id: Optional[str] = None) -> Dict:
        """Helper to create a message with optional tool calls."""
        message = Message(role, content, tool_calls, tool_call_id)
        return message.to_dict()

    def _create_tool_defs(self, raw_tools: List[Any]) -> List[Dict]:
        """Convert raw tool definitions to OpenAI tool format."""
        return [
            {
                "type": "function",
                "function": {
                    "name": t.name,
                    "description": t.description,
                    "parameters": t.inputSchema,
                },
            }
            for t in raw_tools
        ]

    async def _get_tool_defs(self) -> List[Dict[str, Any]]:
        """Get tool definitions, using cache if available."""
        if self.tool_defs is None:
            raw_tools = await retry_on_error(self.mcp.list_tools)
            self.tool_defs = self._create_tool_defs(raw_tools)
            logger.info(f"Loaded {len(self.tool_defs)} tool definitions")
        return self.tool_defs

    @rate_limit
    async def _handle_tool_calls(self, tool_calls: List[Any]) -> None:
        """Handle a sequence of tool calls and update history."""
        for tool_call in tool_calls:
            try:
                # Add this specific tool call to history
                self.history.append(self._add_message(
                    MessageRole.ASSISTANT,
                    None,
                    [tool_call.to_dict()]  # Only add this specific tool call
                ))
                
                # Execute tool call
                tool_name = tool_call.function.name
                kwargs = json.loads(tool_call.function.arguments or "{}")
                
                # Add unique client order ID for stock orders
                if tool_name == "place_stock_order":
                    kwargs['client_order_id'] = str(uuid.uuid4())
                
                logger.info(f"Calling tool: {tool_name} with args: {kwargs}")
                
                result = await retry_on_error(self.mcp.call_tool, tool_name, kwargs)
                print(f"[TOOL CALL] Tool: {tool_name}, Args: {kwargs}")
                
                # Add tool response to history
                response_content = str(result.content)
                self.history.append(self._add_message(
                    MessageRole.TOOL,
                    response_content,
                    tool_call_id=tool_call.id
                ))
                
                # If there was an error in the response, let the LLM know
                if "error" in response_content.lower():
                    logger.warning(f"Tool call returned error: {response_content}")
                    
            except Exception as e:
                error_msg = f"Failed to execute tool {tool_name}: {str(e)}"
                logger.error(error_msg)
                # Add error as tool response
                self.history.append(self._add_message(
                    MessageRole.TOOL,
                    error_msg,
                    tool_call_id=tool_call.id
                ))
                raise ToolCallError(error_msg)

    def _truncate_history(self, history: List[Dict[str, Any]], max_tokens: int = 1000) -> List[Dict[str, Any]]:
        """
        Truncate history to stay within token limits while preserving system message.
        
        Args:
            history: The message history to truncate
            max_tokens: Maximum number of tokens to allow
            
        Returns:
            Truncated history list
        """
        if not history:
            return history
            
        # Always keep the system message
        system_msg = history[0]
        other_msgs = history[1:]
        
        # If we only have the system message, return it
        if not other_msgs:
            return [system_msg]
            
        # Start with the most recent messages
        truncated = []
        current_tokens = 0
        
        # Add messages from newest to oldest until we hit the limit
        for msg in reversed(other_msgs):
            # Rough estimate of tokens (4 chars per token)
            msg_tokens = len(str(msg)) // 4
            if current_tokens + msg_tokens > max_tokens:
                break
            truncated.insert(0, msg)
            current_tokens += msg_tokens
            
        # Add system message back at the start
        return [system_msg] + truncated

    def _get_recent_history(self, max_tokens: int = 300) -> List[Dict[str, Any]]:
        """
        Get the most recent messages from history, always including system message.
        Limits the total tokens to stay within a reasonable range.
        
        Args:
            max_tokens: Maximum number of tokens to allow in history
            
        Returns:
            List of recent messages
        """
        if not self.history:
            return []
            
        # Always include system message
        system_msg = self.history[0]
        system_tokens = len(str(system_msg)) // 4  # Rough estimate
        
        # Start with system message
        result = [system_msg]
        current_tokens = system_tokens
        
        # Add messages from newest to oldest until we hit the limit
        for msg in reversed(self.history[1:]):
            msg_tokens = len(str(msg)) // 4  # Rough estimate
            if current_tokens + msg_tokens > max_tokens:
                break
            result.insert(1, msg)  # Insert after system message
            current_tokens += msg_tokens
            
        return result

    @rate_limit
    async def _openai_chat_with_history(self, history: List[Dict[str, Any]], tools: Optional[List[Dict[str, Any]]] = None):
        """
        Call OpenAI's chat completions API with a specific message history.
        
        Args:
            history: The message history to use
            tools: Optional tool schema definitions to expose.
            
        Returns:
            OpenAI chat completion response.
            
        Raises:
            ChatAgentError: If the API call fails
        """
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=history,
                tools=tools or [],
            )
            
            # Store the token usage
            self.last_token_usage = {
                'prompt_tokens': response.usage.prompt_tokens,
                'completion_tokens': response.usage.completion_tokens,
                'total_tokens': response.usage.total_tokens
            }
            
            # Log token usage from OpenAI's response
            log_token_usage(
                response.usage.prompt_tokens,
                response.usage.completion_tokens,
                response.usage.total_tokens,
                self.model
            )
            
            return response
        except Exception as e:
            logger.error(f"OpenAI API call failed: {str(e)}")
            raise ChatAgentError(f"Failed to get response from OpenAI: {str(e)}")

    @rate_limit
    async def _openai_chat(self, tools: Optional[List[Dict[str, Any]]] = None):
        """Alias for _openai_chat_with_history using the main history."""
        return await self._openai_chat_with_history(self.history, tools)

    @rate_limit
    async def _handle_response(self, msg: Union[ChatCompletionMessage, Any], used_tools: bool = False) -> str:
        """Handle and persist the final response from the model."""
        try:
            if used_tools:
                # OpenAI's create method is synchronous
                oa_response = await self._openai_chat()
                msg = oa_response.choices[0].message
            
            logger.info(f"Received {'tool' if used_tools else 'direct'} response from model")
            print(f"[LLM RESPONSE] {'Tool' if used_tools else 'No tool'} call used.")
            
            self.history.append(self._add_message(MessageRole.ASSISTANT, msg.content))
            return msg.content
        except Exception as e:
            logger.error(f"Failed to handle response: {str(e)}")
            raise ChatAgentError(f"Failed to handle model response: {str(e)}")

    def _print_token_count(self, messages: List[Dict]) -> None:
        """
        Print the token usage from the last LLM response.
        If no response yet, show a message indicating that.
        """
        if self.last_token_usage:
            print(f"\n[TOKEN USAGE]")
            print(f"  Prompt tokens: {self.last_token_usage['prompt_tokens']}")
            print(f"  Completion tokens: {self.last_token_usage['completion_tokens']}")
            print(f"  Total tokens: {self.last_token_usage['total_tokens']}")
            
            # Add context about token usage
            if self.last_token_usage['prompt_tokens'] > 400:
                print("\n[NOTE] High token usage detected. Consider using -d for disposable queries.")
        else:
            print("\n[TOKEN USAGE] No token usage data available yet")

    @disposable_handler
    async def chat(self, user_msg: str, *, disposable: bool = False) -> str:
        """
        Process a user message, handle OpenAI tool-calling, and return a reply.

        Args:
            user_msg: The user's message. Prefix with "-d" for disposable query.
            disposable: If True, do not persist this turn in conversation history.

        Returns:
            str: Assistant's reply.
            
        Raises:
            ChatAgentError: If any part of the chat process fails
        """
        try:
            # Create and update history with user message
            user_message = self._add_message(MessageRole.USER, user_msg)
            temp_history = self._update_history(user_message, not disposable)

            if disposable:
                logger.info("Processing disposable query")
                print("[DISPOSABLE] This Q&A will NOT be persisted.")
            else:
                logger.info("Processing persisted query")
                print("[PERSISTED] This Q&A WILL be saved in history.")

            # Get tool definitions
            tool_defs = await self._get_tool_defs()
            logger.info(f"Using {len(tool_defs)} tools for this request")

            # Get response from OpenAI with full history
            oa_response = await self._openai_chat_with_history(temp_history, tool_defs)
            msg = oa_response.choices[0].message

            # Print token usage after each API call
            self._print_token_count(temp_history)

            # Handle tool calls if present
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                logger.info(f"Tool calls detected: {len(msg.tool_calls)}")
                await self._handle_tool_calls(msg.tool_calls)
                return await self._handle_response(msg, used_tools=True)
            else:
                logger.info("No tool calls in response")
            
            return await self._handle_response(msg, used_tools=False)
            
        except Exception as e:
            logger.error(f"Chat process failed: {str(e)}")
            raise ChatAgentError(f"Chat process failed: {str(e)}")

    @disposable_handler
    async def disposable_chat(self, user_message: str) -> str:
        """
        Process a disposable query that won't be persisted in the conversation history.
        
        Args:
            user_message: The user's message text.
            
        Returns:
            The AI's response text.
            
        Raises:
            ChatAgentError: If the chat processing fails
        """
        try:
            # Create a temporary history with just the system message and current query
            temp_history = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message}
            ]
            
            # Get tool definitions
            tool_defs = await self._get_tool_defs()
            
            # Get response from OpenAI with temporary history
            response = await self._openai_chat_with_history(temp_history, tool_defs)
            
            # Extract and return the response
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"Disposable chat processing failed: {str(e)}")
            raise ChatAgentError(f"Failed to process disposable chat: {str(e)}")
