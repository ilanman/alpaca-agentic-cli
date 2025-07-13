"""
Alpaca MCP Client

Defines AlpacaMCPClient, an asynchronous helper class for communicating
with a running Alpaca MCP server over SSE.

- Wraps connection setup and teardown for easier use with async/await.
- Exposes thin wrappers for listing tools and invoking tool calls.

Intended for use by ChatAgent and related components.

Author: finAI Team
License: MIT
"""

from contextlib import AsyncExitStack
from typing import Any, Dict, Optional, List

from mcp import ClientSession
from mcp.client.sse import sse_client


class AlpacaMCPClient:
    """
    Async context manager for MCP tool listing and invocation.

    Methods:
        list_tools():
            Returns a list of tool definitions available on the server.

        call_tool(name: str, args: dict):
            Invokes a named tool with supplied arguments, returns the result.
    """

    def __init__(self, server_url: str):
        """Initialize the MCP client with the server URL."""
        self.server_url = server_url.rstrip("/") + "/sse"
        self._stack: Optional[AsyncExitStack] = None
        self._session: Optional[ClientSession] = None

    async def __aenter__(self) -> "AlpacaMCPClient":
        """Set up the MCP connection."""
        self._stack = AsyncExitStack()
        await self._stack.__aenter__()

        read_stream, write_fn = await self._stack.enter_async_context(
            sse_client(self.server_url)
        )

        self._session = await self._stack.enter_async_context(
            ClientSession(read_stream, write_fn)
        )

        if self._session:
            await self._session.initialize()
        return self

    async def __aexit__(
        self, exc_type: Optional[type], exc: Optional[Exception], tb: Any
    ) -> None:
        """Clean up the MCP connection."""
        if self._stack:
            await self._stack.aclose()

    async def list_tools(self) -> List[Any]:
        """
        Returns:
            List of tool definitions (as returned by the MCP server).
        """
        if not self._session:
            raise RuntimeError(
                "MCP session not initialized. Use async context manager."
            )

        return (await self._session.list_tools()).tools

    async def call_tool(self, name: str, args: Dict[str, Any]) -> Any:
        """
        Invoke a tool by name with JSON-serializable args.

        Args:
            name: Name of the tool to call.
            args: Dictionary of parameters to send.

        Returns:
            The result from the MCP tool call (server-defined).

        Raises:
            RuntimeError: If session is not initialized.
        """
        if not self._session:
            raise RuntimeError(
                "MCP session not initialized. Use async context manager."
            )

        return await self._session.call_tool(name, args)
