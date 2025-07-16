#!/usr/bin/env python3
"""
Script to start the Alpaca MCP server from the main finAI directory.
"""

from typing import Optional
import subprocess
import sys
import os


def start_alpaca_server(transport: str = "stdio", port: Optional[int] = None):
    """Start the Alpaca MCP server using specified transport."""
    server_dir = os.path.join(os.path.dirname(__file__), "alpaca-mcp-server")

    if not os.path.exists(server_dir):
        print(f"Error: Alpaca MCP server directory not found at {server_dir}")
        print("Please ensure the alpaca-mcp-server directory exists.")
        sys.exit(1)

    print(f"Starting Alpaca MCP Server with transport: {transport}")
    if transport == "stdio":
        print("Server is running with stdio transport (standard MCP protocol)")
        print(
            "This server is designed to be used by MCP clients, not as a standalone HTTP server"
        )
    elif transport == "sse":
        print(f"Server is running with SSE transport on port {port}")
        print("Clients can connect via HTTP to this port using SSE protocol")
    else:
        print(f"Unknown transport: {transport}")
        sys.exit(1)

    print("Press Ctrl+C to stop the server")

    try:
        # Build command to run server with transport and optional port
        cmd = [sys.executable, "alpaca_mcp_server.py", "--transport", transport]
        if port is not None:
            cmd.extend(["--port", str(port)])

        subprocess.run(cmd, cwd=server_dir)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_alpaca_server()
