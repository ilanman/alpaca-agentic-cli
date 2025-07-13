"""
langchain_agent.py

LangChain-based trading agent that enhances the existing system with:
- LangChain's robust tool calling framework
- Additional external data sources (yfinance, news, etc.)
- Better error handling and retry logic
- Streaming responses
- Enhanced memory management

This is designed to work alongside the existing ChatAgent for comparison and gradual migration.
"""

import asyncio
import logging
from typing import List, Dict, Any, Optional
from dataclasses import dataclass

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool, DuckDuckGoSearchRun
from langchain_community.tools import yfinance_tool, WikipediaQueryRun
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import BaseOutputParser

from chat.mcp_client import AlpacaMCPClient

# Configure logging
logging.basicConfig(
    filename='langchain_agent.log',
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ToolResult:
    """Standardized result format for all tools."""
    content: str
    success: bool
    error: Optional[str] = None

class MCPToolWrapper(BaseTool):
    """Wrapper to make MCP tools compatible with LangChain."""
    
    def __init__(self, mcp_client: AlpacaMCPClient, tool_name: str, tool_description: str):
        super().__init__()
        self.mcp_client = mcp_client
        self.tool_name = tool_name
        self.name = f"mcp_{tool_name}"
        self.description = f"MCP Tool: {tool_description}"
    
    async def _arun(self, **kwargs) -> str:
        """Async implementation for MCP tool calls."""
        try:
            result = await self.mcp_client.call_tool(self.tool_name, kwargs)
            return str(result.content)
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.tool_name}: {str(e)}")
            return f"Error: {str(e)}"

class LangChainTradingAgent:
    """
    Enhanced trading agent using LangChain framework.
    
    Features:
    - LangChain's robust tool calling
    - Multiple data sources (Alpaca MCP + external APIs)
    - Conversation memory
    - Streaming responses
    - Better error handling
    """
    
    def __init__(self, mcp_client: AlpacaMCPClient, model: str = "gpt-4o"):
        self.mcp_client = mcp_client
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=0)
        self.memory = ConversationBufferWindowMemory(k=10, return_messages=True)
        self.tools = []
        self.agent_executor = None
        
        # Initialize tools
        self._setup_tools()
        self._setup_agent()
    
    def _setup_tools(self):
        """Set up all available tools (MCP + external)."""
        # Get MCP tools and wrap them
        mcp_tools = asyncio.run(self.mcp_client.list_tools())
        
        for tool in mcp_tools:
            # Create descriptive names for MCP tools
            if "account" in tool.name.lower():
                description = f"Get account information, balances, and portfolio status"
            elif "position" in tool.name.lower():
                description = f"Manage and view trading positions"
            elif "order" in tool.name.lower():
                description = f"Place, cancel, and manage trading orders"
            elif "quote" in tool.name.lower() or "price" in tool.name.lower():
                description = f"Get real-time market data and quotes"
            elif "option" in tool.name.lower():
                description = f"Manage options trading and contracts"
            else:
                description = tool.description
            
            mcp_tool = MCPToolWrapper(self.mcp_client, tool.name, description)
            self.tools.append(mcp_tool)
        
        # Add external tools
        try:
            self.tools.extend([
                DuckDuckGoSearchRun(),
                yfinance_tool,
                WikipediaQueryRun()
            ])
            logger.info("Successfully loaded external tools")
        except Exception as e:
            logger.warning(f"Could not load some external tools: {str(e)}")
    
    def _setup_agent(self):
        """Set up the LangChain agent with tools and memory."""
        # Create a comprehensive system prompt
        system_prompt = """You are an AI trading assistant with access to multiple data sources:

1. **Trading Tools (MCP)**: Real-time trading data, account management, order placement
2. **Research Tools**: Web search, financial data, company information
3. **Analysis Tools**: Technical indicators, fundamental data

When helping users:
- Use trading tools for real-time data and trading actions
- Use research tools for analysis and decision-making
- Combine multiple sources for comprehensive analysis
- Always consider risk management
- Provide clear, actionable recommendations

For trading decisions, explain your reasoning and any risks involved."""

        # Create the prompt template
        prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),
        ])
        
        # Create the agent
        agent = create_openai_functions_agent(
            llm=self.llm,
            tools=self.tools,
            prompt=prompt
        )
        
        # Create the executor
        self.agent_executor = AgentExecutor(
            agent=agent,
            tools=self.tools,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            max_iterations=5
        )
    
    async def chat(self, message: str, stream: bool = False) -> str:
        """
        Process a user message using LangChain agent.
        
        Args:
            message: User's message
            stream: Whether to stream the response
            
        Returns:
            Agent's response
        """
        try:
            logger.info(f"Processing message: {message[:100]}...")
            
            if stream:
                # Streaming response
                response = ""
                async for chunk in self.agent_executor.astream({"input": message}):
                    if "output" in chunk:
                        response += chunk["output"]
                        yield chunk["output"]
                return response
            else:
                # Regular response
                result = await self.agent_executor.ainvoke({"input": message})
                logger.info("Successfully processed message with LangChain agent")
                return result["output"]
                
        except Exception as e:
            error_msg = f"Error in LangChain agent: {str(e)}"
            logger.error(error_msg)
            return error_msg
    
    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        tool_info = {
            "mcp_tools": len([t for t in self.tools if isinstance(t, MCPToolWrapper)]),
            "external_tools": len([t for t in self.tools if not isinstance(t, MCPToolWrapper)]),
            "total_tools": len(self.tools),
            "tool_names": [t.name for t in self.tools]
        }
        return tool_info

# Example usage and testing
async def test_langchain_agent():
    """Test the LangChain agent with sample queries."""
    # Initialize MCP client
    async with AlpacaMCPClient("http://localhost:8000") as mcp:
        # Create LangChain agent
        agent = LangChainTradingAgent(mcp)
        
        # Test queries
        test_queries = [
            "What's my current account balance?",
            "Get the current price of AAPL",
            "Search for recent news about Tesla",
            "What is Tesla's P/E ratio?",
            "Tell me about Apple's business model"
        ]
        
        print("=== LangChain Agent Test ===\n")
        print(f"Available tools: {agent.get_tool_info()}\n")
        
        for query in test_queries:
            print(f"Query: {query}")
            response = await agent.chat(query)
            print(f"Response: {response[:200]}...")
            print("-" * 50)

if __name__ == "__main__":
    asyncio.run(test_langchain_agent()) 