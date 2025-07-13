"""
LangChain Trading Agent

A comprehensive AI trading assistant that combines:
- Real-time trading capabilities via Alpaca MCP
- Multiple data sources (YFinance, AskNews, ArXiv, Reddit)
- Academic research and social sentiment analysis
- Robust error handling and streaming responses

Author: finAI Team
License: MIT
"""

import asyncio
import logging
import warnings
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from pathlib import Path

# Suppress LangChain deprecation warnings
warnings.filterwarnings(
    "ignore", category=DeprecationWarning, module="langchain.memory"
)
warnings.filterwarnings("ignore", message=".*migrating_memory.*")

# Import the specific warning class to filter it
try:
    from langchain_core._api.deprecation import LangChainDeprecationWarning

    warnings.filterwarnings("ignore", category=LangChainDeprecationWarning)
except ImportError:
    pass

import yfinance as yf

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.asknews import AskNewsSearch
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.tools.reddit_search.tool import RedditSearchRun

from chat.mcp_client import AlpacaMCPClient

# Configuration constants
DEFAULT_MODEL = "gpt-4o"
DEFAULT_TEMPERATURE = 0
MEMORY_WINDOW_SIZE = 10
MAX_ITERATIONS = 5
LOG_FILE = Path("langchain_agent.log")

# Configure logging
logging.basicConfig(
    filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
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

    mcp_client: AlpacaMCPClient
    tool_name: str

    def __init__(
        self, mcp_client: AlpacaMCPClient, tool_name: str, tool_description: str
    ):
        super().__init__(
            name=f"mcp_{tool_name}",
            description=f"MCP Tool: {tool_description}",
            mcp_client=mcp_client,
            tool_name=tool_name,
        )

    async def _arun(self, **kwargs) -> str:
        """Async implementation for MCP tool calls with enhanced validation."""
        try:
            # Validate order placement parameters
            if "place_stock_order" in self.tool_name:
                validation_result = self._validate_stock_order_params(kwargs)
                if validation_result:
                    return validation_result

            result = await self.mcp_client.call_tool(self.tool_name, kwargs)
            return str(result.content)
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.tool_name}: {str(e)}")
            return f"Error calling {self.tool_name}: {str(e)}"

    def _validate_stock_order_params(self, kwargs: Dict[str, Any]) -> Optional[str]:
        """Validate stock order parameters."""
        required_params = ["symbol", "side", "quantity"]
        missing_params = [
            param
            for param in required_params
            if param not in kwargs or kwargs[param] is None
        ]

        if missing_params:
            return f"Error: Missing required parameters: {', '.join(missing_params)}. Required: symbol, side (buy/sell), quantity."

        # Validate quantity
        if not isinstance(kwargs.get("quantity"), (int, float)):
            return (
                f"Error: Quantity must be a number, got {type(kwargs.get('quantity'))}"
            )

        # Validate side
        if kwargs.get("side", "").lower() not in ["buy", "sell"]:
            return f"Error: Side must be 'buy' or 'sell', got '{kwargs.get('side')}'"

        return None

    def _run(self, **kwargs) -> str:
        raise NotImplementedError("Synchronous execution is not supported. Use async.")


class YFinanceTool(BaseTool):
    """Custom tool for getting stock data using yfinance library."""

    name: str = "yfinance_stock_data"
    description: str = (
        "Get stock information including price, P/E ratio, market cap, and other financial data for any stock symbol"
    )

    def _run(self, symbol: str) -> str:
        """Get comprehensive stock data for a given symbol."""
        try:
            stock = yf.Ticker(symbol)
            info = stock.info

            # Format market cap and volume with proper handling of None values
            market_cap = info.get("marketCap")
            volume = info.get("volume")

            result = f"""
Stock Information for {symbol.upper()}:
=====================================
Current Price: ${info.get('currentPrice', 'N/A')}
Previous Close: ${info.get('previousClose', 'N/A')}
Market Cap: ${market_cap:,} if market_cap else 'N/A'
P/E Ratio: {info.get('trailingPE', 'N/A')}
52 Week High: ${info.get('fiftyTwoWeekHigh', 'N/A')}
52 Week Low: ${info.get('fiftyTwoWeekLow', 'N/A')}
Volume: {volume:,} if volume else 'N/A'
Company: {info.get('longName', 'N/A')}
Sector: {info.get('sector', 'N/A')}
Industry: {info.get('industry', 'N/A')}
            """.strip()

            return result
        except Exception as e:
            logger.error(f"Error getting stock data for {symbol}: {str(e)}")
            return f"Error getting stock data for {symbol}: {str(e)}"

    async def _arun(self, symbol: str) -> str:
        """Async version of the tool."""
        return self._run(symbol)


class Agent:
    """
    Enhanced trading agent using LangChain framework.

    Features:
    - LangChain's robust tool calling
    - Multiple data sources (Alpaca MCP + external APIs)
    - Conversation memory
    - Streaming responses
    - Better error handling
    """

    def __init__(self, mcp_client: AlpacaMCPClient, model: str = DEFAULT_MODEL):
        self.mcp_client = mcp_client
        self.model = model
        self.llm = ChatOpenAI(model=model, temperature=DEFAULT_TEMPERATURE)
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE, return_messages=True, memory_key="history"
        )
        self.tools: List[BaseTool] = []
        self.agent_executor: Optional[AgentExecutor] = None

    @classmethod
    async def create(
        cls, mcp_client: AlpacaMCPClient, model: str = DEFAULT_MODEL
    ) -> "Agent":
        """Async factory method to create and initialize the agent."""
        logger.info("Creating LangChain agent...")
        self = cls(mcp_client, model)
        logger.info("Agent created, setting up tools...")
        await self._setup_tools()
        logger.info("Tools set up, setting up agent...")
        self._setup_agent()
        logger.info("Agent setup complete")
        return self

    async def _setup_tools(self) -> None:
        """Set up all available tools (MCP + external)."""
        # Setup MCP tools
        await self._setup_mcp_tools()

        # Setup external tools
        await self._setup_external_tools()

    async def _setup_mcp_tools(self) -> None:
        """Set up MCP trading tools."""
        try:
            logger.info("Fetching MCP tools...")
            mcp_tools = await self.mcp_client.list_tools()
            logger.info(f"Found {len(mcp_tools)} MCP tools")
        except Exception as e:
            logger.error(f"Error fetching MCP tools: {str(e)}")
            mcp_tools = []

        for tool in mcp_tools:
            description = self._get_mcp_tool_description(tool.name)
            mcp_tool = MCPToolWrapper(self.mcp_client, tool.name, description)
            self.tools.append(mcp_tool)

    def _get_mcp_tool_description(self, tool_name: str) -> str:
        """Get descriptive name for MCP tool based on its name."""
        tool_name_lower = tool_name.lower()

        if "account" in tool_name_lower:
            return "Get account information, balances, and portfolio status"
        elif "position" in tool_name_lower:
            return "Manage and view trading positions"
        elif "place_stock_order" in tool_name:
            return "Place stock orders. Required: symbol (string), side (buy/sell), quantity (number). Optional: order_type (market/limit/stop), time_in_force (day/gtc), limit_price, stop_price"
        elif "place_option_market_order" in tool_name:
            return "Place option orders. Required: legs (array of option contracts with symbol, side, ratio_qty)"
        elif "cancel" in tool_name_lower:
            return "Cancel trading orders"
        elif "order" in tool_name_lower:
            return "Manage trading orders"
        elif "quote" in tool_name_lower or "price" in tool_name_lower:
            return "Get real-time market data and quotes"
        elif "option" in tool_name_lower:
            return "Manage options trading and contracts"
        else:
            return tool_name

    async def _setup_external_tools(self) -> None:
        """Set up external data source tools."""
        failed_tools = []

        try:
            external_tools = [
                DuckDuckGoSearchRun(),
                YFinanceTool(),
                AskNewsSearch(),
                ArxivQueryRun(),
                RedditSearchRun(),
            ]
            self.tools.extend(external_tools)
            logger.info("Successfully loaded external tools")
        except Exception as e:
            logger.warning(f"Could not load some external tools: {str(e)}")
            print(f"⚠️  Warning: Could not load some external tools: {str(e)}")
            failed_tools.append(str(e))

        if failed_tools:
            print(
                "The following external tools failed to load due to missing API keys or configuration:"
            )
            for err in failed_tools:
                print(" -", err)

    def _setup_agent(self) -> None:
        """Set up the LangChain agent with tools and memory."""
        try:
            logger.info(f"Setting up agent with {len(self.tools)} tools")

            system_prompt = self._get_system_prompt()

            # Create the prompt template
            prompt = ChatPromptTemplate.from_messages(
                [
                    ("system", system_prompt),
                    MessagesPlaceholder(variable_name="history"),
                    ("human", "{input}"),
                    MessagesPlaceholder(variable_name="agent_scratchpad"),
                ]
            )

            # Create the agent
            agent = create_openai_functions_agent(
                llm=self.llm, tools=self.tools, prompt=prompt
            )

            # Create the executor
            self.agent_executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                memory=self.memory,
                verbose=False,
                handle_parsing_errors=True,
                max_iterations=MAX_ITERATIONS,
            )
            logger.info("Successfully created agent executor")
        except Exception as e:
            logger.error(f"Error setting up agent: {str(e)}")
            self.agent_executor = None

    def _get_system_prompt(self) -> str:
        """Get the system prompt for the agent."""
        return """You are an AI trading assistant with access to multiple data sources:

1. **Trading Tools (MCP)**: Real-time trading data, account management, order placement
2. **Financial Data Tools**: yfinance (stock quotes, P/E ratios, market cap), AskNews (financial news), Web search for market research
3. **Research Tools**: ArXiv (academic papers, technical analysis, quantitative finance research)
4. **Social Sentiment**: Reddit (community sentiment, stock discussions, market opinions)
5. **Analysis Tools**: Technical indicators, fundamental data

When helping users:
- Use trading tools for real-time data and trading actions
- For stock orders, use place_stock_order with required parameters: symbol (string), side (buy/sell), quantity (number)
- Use yfinance for stock quotes, P/E ratios, market cap, and financial data
- Use AskNews for real-time financial news and market sentiment
- Use ArXiv for academic research, quantitative finance papers, and technical analysis
- Use Reddit for community sentiment, stock discussions, and market opinions
- Use web search for comprehensive market research
- Combine multiple sources for comprehensive analysis
- Always consider risk management
- Provide clear, actionable recommendations

For trading decisions, explain your reasoning and any risks involved."""

    async def chat(self, message: str) -> str:
        """Process a user message using LangChain agent (non-streaming)."""
        if not self.agent_executor:
            return "Error: Agent not properly initialized"

        try:
            logger.info(f"Processing message: {message[:100]}...")
            result = await self.agent_executor.ainvoke({"input": message})
            logger.info("Successfully processed message with LangChain agent")
            return result["output"]
        except Exception as e:
            error_msg = f"Error in LangChain agent: {str(e)}"
            logger.error(error_msg)
            return error_msg

    async def chat_stream(self, message: str):
        """Streaming version: yields each chunk as it arrives."""
        if not self.agent_executor:
            yield "Error: Agent not properly initialized"
            return

        try:
            logger.info(f"Processing message (streaming): {message[:100]}...")
            async for chunk in self.agent_executor.astream({"input": message}):
                if "output" in chunk:
                    yield chunk["output"]
        except Exception as e:
            error_msg = f"Error in LangChain agent: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def get_tool_info(self) -> Dict[str, Any]:
        """Get information about available tools."""
        return {
            "mcp_tools": len([t for t in self.tools if isinstance(t, MCPToolWrapper)]),
            "external_tools": len(
                [t for t in self.tools if not isinstance(t, MCPToolWrapper)]
            ),
            "total_tools": len(self.tools),
            "tool_names": [t.name for t in self.tools],
        }


async def test_langchain_agent():
    """Test the LangChain agent with sample queries."""
    async with AlpacaMCPClient("http://localhost:8000") as mcp:
        try:
            agent = await Agent.create(mcp)

            test_queries = [
                "What's my current account balance?",
                "Get the current price of AAPL using yfinance",
                "Search the web for information about Apple's latest earnings",
                "Get Tesla's P/E ratio and market cap using yfinance",
                "What is the current market sentiment for TSLA?",
            ]

            print("=== LangChain Agent Test ===\n")
            print(
                f"✅ Loaded {agent.get_tool_info()['total_tools']} tools ({agent.get_tool_info()['mcp_tools']} MCP + {agent.get_tool_info()['external_tools']} external)\n"
            )

            for i, query in enumerate(test_queries, 1):
                print(f"🔍 Test {i}: {query}")
                print("-" * 60)

                try:
                    response = await agent.chat(query)
                    # Clean up the response
                    if "Response:" in response:
                        response = response.split("Response:")[-1].strip()
                    if "> Finished chain." in response:
                        response = response.split("> Finished chain.")[0].strip()

                    # Truncate very long responses
                    if len(response) > 500:
                        response = response[:500] + "..."

                    print(f"✅ Result: {response}")
                except Exception as e:
                    print(f"❌ Error: {str(e)}")

                print("\n" + "=" * 60 + "\n")
        finally:
            if hasattr(agent, "mcp_client"):
                await agent.mcp_client.__aexit__(None, None, None)


if __name__ == "__main__":
    asyncio.run(test_langchain_agent())
