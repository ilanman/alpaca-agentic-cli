"""
finAI Trading Agent - Main Entry Point

Enhanced main entry point using LangChain.
Provides a modern trading agent with multiple data sources and improved
capabilities.

Usage:
    $ python main.py              # Run LangChain agent
    $ python main.py --test       # Run test queries

Author: finAI Team
License: MIT
"""

import asyncio
import os
import argparse
import warnings
import sys

from dotenv import load_dotenv
from chat.mcp_client import AlpacaMCPClient
from chat.agent import Agent

# Suppress ResourceWarnings about unclosed transports
warnings.filterwarnings(
    "ignore", category=ResourceWarning, message="unclosed transport"
)

# Configuration constants
DEFAULT_SERVER_URL = "http://localhost:8000"
DEFAULT_MODEL = "gpt-4o"
TEST_QUERIES = [
    "What's my account balance?",
    "Get the current price of AAPL",
    "Search for recent news about Tesla",
    "What is Tesla's P/E ratio?",
    "Tell me about Apple's business model",
]


async def run_langchain_agent() -> None:
    """Run the enhanced LangChain agent in interactive mode."""
    server_url = os.getenv("MCP_SERVER_URL", DEFAULT_SERVER_URL)
    print(f"🔌 Connecting to MCP at {server_url}...")

    async with AlpacaMCPClient(server_url) as mcp:
        try:
            agent = await Agent.create(mcp)

            # Show available tools
            tool_info = agent.get_tool_info()
            print("🤖 LangChain Trading Agent Ready!")
            print(f"📊 Available tools: {tool_info['total_tools']} total")
            print(f"   - MCP tools: {tool_info['mcp_tools']}")
            print(f"   - External tools: {tool_info['external_tools']}")
            print("Type 'exit' to quit.\n")

            while True:
                try:
                    user = input("You: ").strip()
                    if user.lower() in {"exit", "quit"}:
                        break

                    async for chunk in agent.chat_stream(user):
                        print(chunk, end="", flush=True)
                    print("\n")
                except KeyboardInterrupt:
                    break
                except Exception as exc:
                    print(f"⚠️  Error: {exc}")
        finally:
            # Clean up resources
            await agent.mcp_client.__aexit__(None, None, None)


async def run_test() -> None:
    """Run test queries to verify functionality."""
    server_url = os.getenv("MCP_SERVER_URL", DEFAULT_SERVER_URL)

    print("🧪 Running LangChain Agent Test\n")

    async with AlpacaMCPClient(server_url) as mcp:
        try:
            agent = await Agent.create(mcp)

            # Show tool info
            tool_info = agent.get_tool_info()
            print(f"Available tools: {tool_info['total_tools']} total")
            print(f"  - MCP tools: {tool_info['mcp_tools']}")
            print(f"  - External tools: {tool_info['external_tools']}\n")

            for query in TEST_QUERIES:
                print(f"Query: {query}")
                try:
                    response = await agent.chat(query)
                    print(f"Response: {response[:200]}...")
                except Exception as e:
                    print(f"Error: {e}")
                print("-" * 50)
        finally:
            if hasattr(agent, "mcp_client"):
                await agent.mcp_client.__aexit__(None, None, None)


def main() -> None:
    """Main entry point for the application."""
    # Load environment variables
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="finAI Trading Agent with LangChain Integration"
    )
    parser.add_argument(
        "--test",
        action="store_true",
        help="Run test queries to verify functionality",
    )

    args = parser.parse_args()

    try:
        if args.test:
            asyncio.run(run_test())
        else:
            asyncio.run(run_langchain_agent())
    except KeyboardInterrupt:
        print("\n👋 Goodbye!")
    except Exception as e:
        print(f"❌ Fatal error: {e}")
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    main()
