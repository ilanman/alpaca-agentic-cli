"""
chat/mcp_client.py

Defines AlpacaMCPClient, an asynchronous helper class for communicating
with a running Alpaca MCP server over SSE.

- Wraps connection setup and teardown for easier use with async/await.
- Exposes thin wrappers for listing tools and invoking tool calls.

Intended for use by ChatAgent and related components.
"""

from contextlib import AsyncExitStack
from typing import Any, Dict, Optional

from mcp import ClientSession
from mcp.client.sse import sse_client   # SSE transport

class AlpacaMCPClient:
    """
    Async context manager for MCP tool listing and invocation.

    Methods:
        list_tools():
            Returns a list of tool definitions currently available on the server.

        call_tool(name: str, args: dict):
            Invokes a named tool with supplied arguments, returns the tool result.
    """
        
    def __init__(self, server_url: str):
        self.server_url = server_url.rstrip("/") + "/sse"
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    async def __aenter__(self):
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()
        read_stream, write_fn = await self._stack.enter_async_context(
            sse_client(self.server_url)
        )
        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_fn)
        )
        await self._session.initialize()
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._stack:
            await self._stack.aclose()

    async def list_tools(self):
        """
        Returns:
            List of tool definitions (as returned by the MCP server).
        """
        return (await self._session.list_tools()).tools

    async def call_tool(self, name: str, args: Dict[str, Any]):
        """
        Invoke a tool by name with JSON-serializable args.

        Args:
            name: Name of the tool to call.
            args: Dictionary of parameters to send.

        Returns:
            The result from the MCP tool call (server-defined).
        """
        return await self._session.call_tool(name, args)
