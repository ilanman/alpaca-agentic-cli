#!/usr/bin/env python3
"""
Script to start the Alpaca MCP server from the main finAI directory.
"""

import subprocess
import sys
import os


def start_alpaca_server():
    """Start the Alpaca MCP server using stdio transport."""
    server_dir = os.path.join(os.path.dirname(__file__), "alpaca-mcp-server")

    if not os.path.exists(server_dir):
        print(f"Error: Alpaca MCP server directory not found at {server_dir}")
        print("Please ensure the alpaca-mcp-server directory exists.")
        sys.exit(1)

    print("Starting Alpaca MCP Server...")
    print("Server is running with stdio transport (standard MCP protocol)")
    print(
        "This server is designed to be used by MCP clients, not as a standalone HTTP server"
    )
    print("Press Ctrl+C to stop the server")

    try:
        # Change to the server directory and run the main server script
        subprocess.run([sys.executable, "alpaca_mcp_server.py"], cwd=server_dir)
    except KeyboardInterrupt:
        print("\nServer stopped.")
    except Exception as e:
        print(f"Error starting server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    start_alpaca_server()
