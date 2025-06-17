"""
main.py

Command-line entry point for the Alpaca Agentic Trading CLI.

- Loads environment variables and initializes the ChatAgent.
- Connects to the Alpaca MCP server via MCP client.
- Handles user input loop, supports disposable ("-d") queries.
- Prints token counts and tool call logs for every interaction.

Usage:
    $ python main.py
"""

import asyncio
import os
import signal
import sys
import logging
from typing import Optional
from dotenv import load_dotenv
from chat.agent import ChatAgent, ChatAgentError
from chat.mcp_client import AlpacaMCPClient

load_dotenv()

# Configure logger
logger = logging.getLogger(__name__)

# Configuration
DEFAULT_SERVER_URL = "http://localhost:8000"
EXIT_COMMANDS = {"exit", "quit", "q"}

class GracefulExit(SystemExit):
    """Custom exception for graceful program termination."""
    pass

def signal_handler(signum, frame):
    """Handle termination signals gracefully."""
    raise GracefulExit()

async def get_user_input(prompt: str = "You: ") -> Optional[str]:
    """Get user input with proper error handling."""
    try:
        return input(prompt).strip()
    except (EOFError, KeyboardInterrupt):
        return None

async def handle_chat_loop(agent: ChatAgent) -> None:
    """Handle the main chat interaction loop."""
    while True:
        try:
            user_input = await get_user_input()
            if not user_input:
                print("\nGoodbye!")
                break
                
            if user_input.lower() in EXIT_COMMANDS:
                print("\nGoodbye!")
                break
                
            # Handle special commands
            if user_input.lower() == "erase history":
                agent.erase_history()
                print("\nAI: History has been erased. How can I help you?\n")
                continue
                
            reply = await agent.chat(user_input)
            print(f"\nAI: {reply}\n")
            
        except ChatAgentError as e:
            print(f"⚠️  Error: {e}")
        except Exception as e:
            print(f"⚠️  Unexpected error: {e}")
            logger.error(f"Unexpected error in chat loop: {e}", exc_info=True)

async def main():
    """Main entry point for the application."""
    # Set up signal handlers for graceful shutdown
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    server_url = os.getenv("MCP_SERVER_URL", DEFAULT_SERVER_URL)
    print(f"🔌 Connecting to MCP at {server_url} …")

    try:
        async with AlpacaMCPClient(server_url) as mcp:
            agent = ChatAgent(mcp)
            print("🤖 Ready!  Type 'exit' to quit.\n")
            await handle_chat_loop(agent)
            
    except Exception as e:
        print(f"⚠️  Fatal error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except GracefulExit:
        print("\nShutting down gracefully...")
        sys.exit(0)
    except Exception as e:
        print(f"⚠️  Fatal error: {e}")
        sys.exit(1)
