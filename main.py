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

import asyncio, os
from dotenv import load_dotenv

load_dotenv()
from chat.agent import ChatAgent
from chat.mcp_client import AlpacaMCPClient

async def main():
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    print(f"🔌 Connecting to MCP at {server_url} …")

    async with AlpacaMCPClient(server_url) as mcp:
        agent = ChatAgent(mcp)
        print("🤖 Ready!  Type 'exit' to quit.\n")

        while True:
            user = input("You: ").strip()
            if user.lower() in {"exit", "quit"}:
                break
            try:
                reply = await agent.chat(user)
                print(f"\nAI: {reply}\n")
            except KeyboardInterrupt:
                break
            except Exception as exc:
                print(f"⚠️  Error: {exc}")

if __name__ == "__main__":
    asyncio.run(main())
