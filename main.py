"""
main.py

Enhanced main entry point using LangChain integration.
Provides a modern trading agent with multiple data sources and improved capabilities.

Usage:
    $ python main.py              # Run LangChain agent
    $ python main.py --test       # Run test queries
"""

import asyncio
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from chat.mcp_client import AlpacaMCPClient
from langchain_agent import LangChainTradingAgent

async def run_langchain_agent():
    """Run the enhanced LangChain agent."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    print(f"🔌 Connecting to MCP at {server_url} …")

    async with AlpacaMCPClient(server_url) as mcp:
        agent = LangChainTradingAgent(mcp)
        
        # Show available tools
        tool_info = agent.get_tool_info()
        print(f"🤖 LangChain Trading Agent Ready!")
        print(f"📊 Available tools: {tool_info['total_tools']} total")
        print(f"   - MCP tools: {tool_info['mcp_tools']}")
        print(f"   - External tools: {tool_info['external_tools']}")
        print("Type 'exit' to quit.\n")

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

async def run_test():
    """Run test queries to verify functionality."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    
    test_queries = [
        "What's my account balance?",
        "Get the current price of AAPL",
        "Search for recent news about Tesla",
        "What is Tesla's P/E ratio?",
        "Tell me about Apple's business model"
    ]
    
    print("🧪 Running LangChain Agent Test\n")
    
    async with AlpacaMCPClient(server_url) as mcp:
        agent = LangChainTradingAgent(mcp)
        
        # Show tool info
        tool_info = agent.get_tool_info()
        print(f"Available tools: {tool_info['total_tools']} total")
        print(f"  - MCP tools: {tool_info['mcp_tools']}")
        print(f"  - External tools: {tool_info['external_tools']}\n")
        
        for query in test_queries:
            print(f"Query: {query}")
            try:
                response = await agent.chat(query)
                print(f"Response: {response[:200]}...")
            except Exception as e:
                print(f"Error: {e}")
            print("-" * 50)

def main():
    parser = argparse.ArgumentParser(description="finAI Trading Agent with LangChain Integration")
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run test queries to verify functionality"
    )
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(run_test())
    else:
        asyncio.run(run_langchain_agent())

if __name__ == "__main__":
    main() 