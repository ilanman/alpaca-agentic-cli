"""
main_langchain.py

Enhanced main entry point that demonstrates LangChain integration.
Allows users to choose between the original ChatAgent and the new LangChainTradingAgent.

Usage:
    $ python main_langchain.py --agent original    # Use original ChatAgent
    $ python main_langchain.py --agent langchain   # Use LangChain agent (default)
    $ python main_langchain.py --test              # Run comparison tests
"""

import asyncio
import os
import argparse
from dotenv import load_dotenv

load_dotenv()

from chat.agent import ChatAgent
from chat.mcp_client import AlpacaMCPClient
from langchain_agent import LangChainTradingAgent

async def run_original_agent():
    """Run the original ChatAgent."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    print(f"🔌 Connecting to MCP at {server_url} …")

    async with AlpacaMCPClient(server_url) as mcp:
        agent = ChatAgent(mcp)
        print("🤖 Original Agent Ready! Type 'exit' to quit.\n")

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

async def run_langchain_agent():
    """Run the enhanced LangChain agent."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    print(f"🔌 Connecting to MCP at {server_url} …")

    async with AlpacaMCPClient(server_url) as mcp:
        agent = LangChainTradingAgent(mcp)
        
        # Show available tools
        tool_info = agent.get_tool_info()
        print(f"🤖 LangChain Agent Ready!")
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

async def run_comparison_test():
    """Run a comparison test between both agents."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    
    test_queries = [
        "What's my account balance?",
        "Get the current price of AAPL",
        "Search for recent news about Tesla"
    ]
    
    print("🧪 Running Agent Comparison Test\n")
    
    async with AlpacaMCPClient(server_url) as mcp:
        # Test original agent
        print("=== Original ChatAgent ===")
        original_agent = ChatAgent(mcp)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                response = await original_agent.chat(query)
                print(f"Response: {response[:150]}...")
            except Exception as e:
                print(f"Error: {e}")
        
        print("\n" + "="*50)
        
        # Test LangChain agent
        print("=== LangChain TradingAgent ===")
        langchain_agent = LangChainTradingAgent(mcp)
        
        for query in test_queries:
            print(f"\nQuery: {query}")
            try:
                response = await langchain_agent.chat(query)
                print(f"Response: {response[:150]}...")
            except Exception as e:
                print(f"Error: {e}")

def main():
    parser = argparse.ArgumentParser(description="finAI Trading Agent with LangChain Integration")
    parser.add_argument(
        "--agent", 
        choices=["original", "langchain"], 
        default="langchain",
        help="Choose which agent to use (default: langchain)"
    )
    parser.add_argument(
        "--test", 
        action="store_true",
        help="Run comparison test between agents"
    )
    
    args = parser.parse_args()
    
    if args.test:
        asyncio.run(run_comparison_test())
    elif args.agent == "original":
        asyncio.run(run_original_agent())
    else:
        asyncio.run(run_langchain_agent())

if __name__ == "__main__":
    main() 