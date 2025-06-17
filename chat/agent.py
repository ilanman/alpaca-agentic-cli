"""
chat/agent.py

Defines the ChatAgent class, responsible for managing the conversation loop,
OpenAI LLM integration, MCP tool-calling, disposable queries, and token tracking.

Key features:
    - OpenAI function/tool calling support
    - Disposable (non-persistent) query mode with "-d" or "--disposable" prefix
    - Per-turn token window logging (cost control)
    - Tool call logging for transparency
    - Robust OpenAI tool/response protocol compliance

See README.md for architecture overview and example usage.
"""

import json
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from openai import OpenAI
from .mcp_client import AlpacaMCPClient

# Configure logging
logging.basicConfig(
    filename='chat_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an AI trading assistant. "
    "When appropriate, call MCP tools to fetch data or place orders."
)

@dataclass
class Message:
    """Represents a message in the conversation history."""
    role: str
    content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_call_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert message to dictionary format for API calls."""
        msg_dict = {"role": self.role}
        if self.content is not None:
            msg_dict["content"] = self.content
        if self.tool_calls is not None:
            msg_dict["tool_calls"] = self.tool_calls
        if self.tool_call_id is not None:
            msg_dict["tool_call_id"] = self.tool_call_id
        return msg_dict

def disposable_handler(chat_method):
    """
    Decorator to handle disposable query mode.
    
    Args:
        chat_method: The chat method to wrap
        
    Returns:
        Wrapped method that handles disposable queries
    """
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
        mcp (AlpacaMCPClient): Async client to the running MCP server
        model (str): OpenAI model name (e.g. "gpt-4o")
        history (List[Message]): Persisted conversation history
        openai (OpenAI): OpenAI client instance
        cumulative_tokens (Dict[str, int]): Running total of tokens used in the session
    """

    def __init__(self, mcp: AlpacaMCPClient, model: str = "gpt-4o"):
        """
        Initialize the ChatAgent.
        
        Args:
            mcp: AlpacaMCPClient instance
            model: OpenAI model name
        """
        self.mcp = mcp
        self.model = model
        self.openai = OpenAI()
        self.history: List[Message] = [Message(role="system", content=SYSTEM_PROMPT)]
        self.cumulative_tokens = {
            "prompt": 0,
            "completion": 0,
            "total": 0
        }
        logger.info(f"Initialized ChatAgent with model {model}")

    def _log_token_usage(self, response: Any) -> None:
        """
        Log token usage from OpenAI API response and update cumulative counts.
        
        Args:
            response: OpenAI API response containing token usage information
        """
        if hasattr(response, 'usage'):
            usage = response.usage
            
            # Update cumulative counts
            self.cumulative_tokens["prompt"] += usage.prompt_tokens
            self.cumulative_tokens["completion"] += usage.completion_tokens
            self.cumulative_tokens["total"] += usage.total_tokens
            
            # Log to file
            logger.info(f"Token usage - Prompt: {usage.prompt_tokens}, "
                       f"Completion: {usage.completion_tokens}, "
                       f"Total: {usage.total_tokens}")
            logger.info(f"Cumulative tokens - Prompt: {self.cumulative_tokens['prompt']}, "
                       f"Completion: {self.cumulative_tokens['completion']}, "
                       f"Total: {self.cumulative_tokens['total']}")
            
            # Print to terminal
            print("\n[TOKEN USAGE]")
            print(f"Current request (including full context history):")
            print(f"  Prompt tokens: {usage.prompt_tokens}")
            print(f"  Completion tokens: {usage.completion_tokens}")
            print(f"  Total tokens: {usage.total_tokens}")
            print(f"\nSession totals (sum of all API calls):")
            print(f"  Prompt tokens: {self.cumulative_tokens['prompt']}")
            print(f"  Completion tokens: {self.cumulative_tokens['completion']}")
            print(f"  Total tokens: {self.cumulative_tokens['total']}")

    async def _openai_chat(self, tools: Optional[List[Dict[str, Any]]] = None) -> Any:
        """
        Call OpenAI's chat completions API with the current message history.
        
        Args:
            tools: Optional list of tool schema definitions
            
        Returns:
            OpenAI chat completion response
        """
        messages = [msg.to_dict() for msg in self.history]
        
        try:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=messages,
                tools=tools or [],
            )
            self._log_token_usage(response)
            logger.info("Successfully received response from OpenAI")
            return response
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {str(e)}")
            raise

    async def _handle_tool_calls(self, tool_calls: List[Any]) -> None:
        """
        Handle tool calls from the LLM response.
        
        Args:
            tool_calls: List of tool calls to execute
        """
        for tool_call in tool_calls:
            tool_name = tool_call.function.name
            kwargs = json.loads(tool_call.function.arguments or "{}")
            
            logger.info(f"Executing tool call: {tool_name} with args: {kwargs}")
            result = await self.mcp.call_tool(tool_name, kwargs)
            
            self.history.append(Message(
                role="tool",
                tool_call_id=tool_call.id,
                content=str(result.content)
            ))

    @disposable_handler
    async def chat(self, user_msg: str, *, disposable: bool = False) -> str:
        """
        Process a user message, handle OpenAI tool-calling, and return a reply.
        
        Args:
            user_msg: The user's message
            disposable: If True, do not persist this turn in conversation history
            
        Returns:
            Assistant's reply
        """
        logger.info(f"Processing message (disposable={disposable}): {user_msg[:100]}...")
        
        # Add user message to history
        user_message = Message(role="user", content=user_msg)
        if not disposable:
            self.history.append(user_message)
        
        # Get available tools
        raw_tools = await self.mcp.list_tools()
        tool_defs = [
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
        
        # Get initial response from OpenAI
        oa_response = await self._openai_chat(tool_defs)
        msg = oa_response.choices[0].message
        
        # Handle tool calls if present
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            print("\n[RESPONSE SOURCE] Generated using LLM with tool assistance")
            logger.info("Processing tool calls from LLM response")
            self.history.append(Message(
                role="assistant",
                tool_calls=[tc.to_dict() for tc in msg.tool_calls]
            ))
            
            await self._handle_tool_calls(msg.tool_calls)
            
            # Get final response with tool outputs
            oa_response = await self._openai_chat()
            msg = oa_response.choices[0].message
        else:
            print("\n[RESPONSE SOURCE] Generated directly from LLM without tools")
        
        # Add assistant's response to history
        if not disposable:
            self.history.append(Message(role="assistant", content=msg.content))
        
        logger.info("Successfully processed message and generated response")
        return msg.content
