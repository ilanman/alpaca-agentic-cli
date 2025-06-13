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
from typing import Dict, List, Any

from openai import OpenAI
from .mcp_client import AlpacaMCPClient

SYSTEM_PROMPT = (
    "You are an AI trading assistant. "
    "When appropriate, call MCP tools to fetch data or place orders."
)

def count_tokens(messages, model="gpt-4o"):
    try:
        enc = tiktoken.encoding_for_model(model)
    except KeyError:
        enc = tiktoken.get_encoding("cl100k_base")  # fallback for unknown models

    total = 0
    for m in messages:
        # The format is: {"role": "...", "content": "...", ...}
        for k, v in m.items():
            if isinstance(v, str):
                total += len(enc.encode(v))
            elif isinstance(v, list):
                for item in v:
                    if isinstance(item, dict):
                        for v2 in item.values():
                            if isinstance(v2, str):
                                total += len(enc.encode(v2))
    return total

def disposable_handler(chat_method):
    async def wrapper(self, user_msg, *args, **kwargs):
        disposable = False
        if user_msg.startswith("-d "):
            disposable = True
            user_msg = user_msg[3:].strip()
        elif user_msg.startswith("--disposable "):
            disposable = True
            user_msg = user_msg[len("--disposable "):].strip()
        # Pass 'disposable' flag as a kwarg
        return await chat_method(self, user_msg, *args, disposable=disposable, **kwargs)
    return wrapper

class ChatAgent:
    """
    Conversational agent integrating OpenAI LLM with Alpaca MCP tool calls.

    Attributes:
        mcp (AlpacaMCPClient): Async client to the running MCP server.
        model (str): OpenAI model name (e.g. "gpt-4o").
        history (List[Dict]): Persisted conversation history (user/assistant/tool messages).

    Methods:
        chat(user_msg: str, *, disposable=False) -> str:
            Process user input and return LLM response, handling tools as needed.

        _openai_chat(...):
            Internal utility to call OpenAI API with the constructed message history.

        _print_token_count(messages):
            Print cumulative token count for supplied message history (for cost awareness).
    """
    
    def __init__(self, mcp: AlpacaMCPClient, model: str = "gpt-4o"):
        self.mcp = mcp
        self.model = model
        self.openai = OpenAI()
        self.history: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT}
        ]

    async def _openai_chat(self, tools: List[Dict[str, Any]] = None):
        """
        Call OpenAI's chat completions API with the current or override message history.

        Args:
            tools (List[dict], optional): Tool schema definitions to expose.
            override_history (List[dict], optional): Use this instead of self.history.

        Returns:
            OpenAI chat completion response.
        """
                
        return self.openai.chat.completions.create(
            model=self.model,
            messages=self.history,
            tools=tools or [],
        )
    
    # Add to your ChatAgent class:
    def _print_token_count(self, messages):
        """
        Print the number of tokens in the supplied message list.

        Args:
            messages (List[dict]): Message history to count tokens for.
        """
        
        tokens = count_tokens(messages, self.model)
        print(f"[TOKENS] Cumulative token window being sent: {tokens} tokens")

    @disposable_handler
    async def chat(self, user_msg: str, *, disposable=False) -> str:
        """
        Process a user message, handle OpenAI tool-calling, and return a reply.

        Args:
            user_msg (str): The user's message. Prefix with "-d" for disposable query.
            disposable (bool): If True, do not persist this turn in conversation history.

        Returns:
            str: Assistant's reply.
        """

        if disposable:
            # Only use the current persisted history + THIS user message (don't persist this exchange after)
            temp_history = self.history + [{"role": "user", "content": user_msg}]
        else:
            temp_history = self.history + [{"role": "user", "content": user_msg}]
            self.history.append({"role": "user", "content": user_msg})  # Only persist here!

        self._print_token_count(temp_history)

        if disposable:
            print("[DISPOSABLE] This Q&A will NOT be persisted.")
        else:
            print("[PERSISTED] This Q&A WILL be saved in history.")

        # Discover available tools...
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

        # Print token count for the current history
        current_messages = temp_history if disposable else self.history
        self._print_token_count(current_messages)

        # 1. Send user message + tool schema, get LLM response
        oa_response = await self._openai_chat(tool_defs)
        msg = oa_response.choices[0].message

        # 2. If model wants to call tools, handle it
        if hasattr(msg, "tool_calls") and msg.tool_calls:
            # Add the assistant's tool_calls message to history (content must be None for tool calls)
            self.history.append({
                "role": "assistant",
                "content": None,
                "tool_calls": [tc.to_dict() for tc in msg.tool_calls],  # Ensure dict format for API
            })
            # For each tool call, execute and add a tool response to history
            for tool_call in msg.tool_calls:
                tool_name = tool_call.function.name
                kwargs = json.loads(tool_call.function.arguments or "{}")
                result = await self.mcp.call_tool(tool_name, kwargs)
                print(f"[TOOL CALL] Tool: {tool_name}, Args: {kwargs}")
                # Add tool's reply to history
                self.history.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": str(result.content),  # Make sure content is serializable
                })
            # 3. Get model's final answer with tool outputs in history (no tools param needed now)
            oa_response = await self._openai_chat()
            msg = oa_response.choices[0].message
            self.history.append({"role": "assistant", "content": msg.content})
            return msg.content
        else:
            print("[LLM RESPONSE] No tool call used. Response came from LLM.")
            self.history.append({"role": "assistant", "content": msg.content})
            return msg.content
