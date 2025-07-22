"""
Trading Agent

A comprehensive AI trading assistant that combines:
- Real-time trading capabilities via Alpaca MCP
- Multiple data sources (YFinance, AskNews, ArXiv, Reddit)
- Academic research and social sentiment analysis
- Robust error handling and streaming responses

Author: finAI Team
License: MIT
"""

import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
import sys


import yfinance as yf  # type: ignore

from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_community.tools.asknews import AskNewsSearch
from langchain_community.tools import DuckDuckGoSearchRun, ArxivQueryRun
from langchain_community.tools.reddit_search.tool import RedditSearchRun
from langchain_experimental.tools import PythonREPLTool
from datetime import datetime, timedelta

from chat.mcp_client import AlpacaMCPClient
from chat.backtest_agent import (
    BacktestTool,
    ListStrategiesTool,
    InputInstructionsTool,
    CompareStrategiesTool,
)

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

                # Only confirm if running in an interactive session
                if sys.stdin.isatty():
                    confirmed = await self._confirm_trade(kwargs)
                    if not confirmed:
                        print("Trade cancelled.")
                        return "Trade cancelled by user."

            print(
                "Trade request made. Awaiting response. This may take a few seconds..."
            )
            result = await self.mcp_client.call_tool(self.tool_name, kwargs)
            return str(result.content)
        except RuntimeError as e:
            # Detect MCP session not initialized error and return user-friendly message
            if "MCP session not initialized" in str(e):
                return (
                    "MCP server is not connected. "
                    "Cannot execute trade or access MCP tools."
                )
            logger.error(f"Error calling MCP tool {self.tool_name}: {str(e)}")
            return f"Error calling {self.tool_name}: {str(e)}"
        except Exception as e:
            logger.error(f"Error calling MCP tool {self.tool_name}: {str(e)}")
            return f"Error calling {self.tool_name}: {str(e)}"

    async def _confirm_trade(self, kwargs: Dict[str, Any]) -> bool:
        """Ask the user to confirm the trade, showing details and live price."""
        symbol = kwargs.get("symbol")
        if not symbol:
            print("Error: Symbol is required for trade confirmation.")
            return False
        side = kwargs.get("side")
        quantity = kwargs.get("quantity")
        # Fetch current price using yfinance
        try:
            stock = yf.Ticker(symbol)
            info = stock.info
            price = info.get("currentPrice", "N/A")
        except Exception:
            price = "N/A"
        print("\n=== Trade Confirmation ===")
        print(
            f"You are about to {side} {quantity} shares of {symbol.upper()} at ${price} per share."
        )
        if price == "N/A" or quantity is None:
            total = "N/A"
        else:
            try:
                total = str(round(float(price) * float(quantity), 2))
            except Exception:
                total = "N/A"
        print(f"Total: ${total}")
        print(
            "Do you want to proceed? [yes/y/confirm]: \nAny other key will cancel the trade.",
            end="",
            flush=True,
        )
        resp = input().strip().lower()
        return resp in ("y", "yes", "confirm")

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
    """
    Custom tool for getting stock data using yfinance.
    Can fetch current real-time data or a single historical closing price.
    """

    name: str = "get_stock_data"
    description: str = (
        "Get stock data. For real-time data, provide just the 'symbol'. "
        "For a historical closing price, provide the 'symbol' and a 'date' in 'YYYY-MM-DD' format."
    )

    def _run(self, symbol: str, date: Optional[str] = None) -> str:
        """Get stock data. If date is provided, gets historical close. Otherwise, gets current data."""
        if date:
            return self._get_historical_price(symbol, date)
        else:
            return self._get_current_info(symbol)

    def _get_historical_price(self, symbol: str, date: str) -> str:
        """Get the closing price for a stock on a specific date."""
        try:
            # Validate date format
            parsed_date = datetime.strptime(date, "%Y-%m-%d")
            # yfinance needs the day after for the end date to include the requested date
            end_date = parsed_date + timedelta(days=1)
            data = yf.download(
                symbol,
                start=date,
                end=end_date.strftime("%Y-%m-%d"),
                progress=False,
            )
            if data.empty:
                return f"Error: No data found for {symbol} on or before {date}. The market may have been closed. Please try an earlier date."
            price = data["Close"].iloc[0]
            return f"The closing price for {symbol.upper()} on {date} was ${price:.2f}"
        except Exception as e:
            logger.error(f"Error getting historical price for {symbol} on {date}: {e}")
            return f"Error getting historical price for {symbol} on {date}: {e}"

    def _get_current_info(self, symbol: str) -> str:
        """Get comprehensive current stock data for a given symbol."""
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

    async def _arun(self, symbol: str, date: Optional[str] = None) -> str:
        """Async version of the tool."""
        return self._run(symbol, date)


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
        self.llm = ChatOpenAI(
            model=model, temperature=DEFAULT_TEMPERATURE, model_kwargs={"stream": True}
        )
        self.memory = ConversationBufferWindowMemory(
            k=MEMORY_WINDOW_SIZE,
            return_messages=True,
            memory_key="history",
            output_key="output",
        )
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
                PythonREPLTool(),
                BacktestTool(),
                ListStrategiesTool(),
                InputInstructionsTool(),
                CompareStrategiesTool(),
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
        return """
You are an AI trading assistant with access to multiple data sources:

1. **Trading Tools (MCP)**: Real-time trading data, account management, order placement
2. **Financial Data Tools**: yfinance (stock quotes, P/E ratios, market cap), AskNews (financial news), Web search for market research
3. **Research Tools**: ArXiv (academic papers, technical analysis, quantitative finance research)
4. **Social Sentiment**: Reddit (community sentiment, stock discussions, market opinions)
5. **Backtesting Tools**: Run historical strategy backtests with multiple strategies (SMA crossover, RSI, Bollinger Bands, Buy & Hold)
6. **Analysis Tools**: Technical indicators, fundamental data

When helping users:
- Use trading tools for real-time data and trading actions
- For stock orders, use place_stock_order with required parameters: symbol (string), side (buy/sell), quantity (number)
- Use yfinance for stock quotes, P/E ratios, market cap, and financial data
- To get a historical stock price, use `get_stock_data` with both `symbol` and `date`.
- To perform calculations or complex comparisons, use the `PythonREPLTool`. You can write Python code to process data from other tools.
- Use ArXiv for academic research, quantitative finance papers, and technical analysis
- Use Reddit for community sentiment, stock discussions, and market opinions
- Use web search for comprehensive market research
- Use backtest tool to test trading strategies on historical data
- Use list_strategies to show available backtesting strategies
- Use input_instructions to get detailed input formats for backtesting strategies
- Use compare_stock_returns to compare performance of multiple stocks over a time period
- Combine multiple sources for comprehensive analysis
- Always consider risk management
- Provide clear, actionable recommendations

**Formatting:** When presenting calculations, use simple, readable language. For example, use 'x' for multiplication, not LaTeX symbols like `\\times`. Use 'is approximately' instead of `\\approx`.

**IMPORTANT:** For any user request mentioning a backtest, strategy name, or historical strategy test, ALWAYS call the backtest tool, even if parameters are missing or incomplete. The backtest tool will provide the user with the exact required format and a copy-pasteable example for each strategy. Do not respond with a conversational prompt for backtesting; let the tool handle all formatting and error messages.

For trading decisions, explain your reasoning and any risks involved.
For backtesting requests, specify the strategy, symbol, date range, and any strategy parameters, or let the backtest tool prompt the user for the correct format.
"""

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
        """Streaming version: yields only LLM/assistant output, with no extra labels or prefixes."""
        if not self.agent_executor:
            yield "Error: Agent not properly initialized"
            return

        try:
            logger.info(f"Processing message (streaming): {message[:100]}...")
            async for event in self.agent_executor.astream_events(
                {"input": message}, version="v1"
            ):
                for ev in self._flatten_events(event):
                    try:
                        if isinstance(ev, dict):
                            event_type = ev.get("event", "")
                            if event_type != "on_chat_model_stream":
                                continue  # Only yield LLM output events
                        else:
                            continue  # Skip non-dict events
                        content = _extract_content(ev)
                        if content:
                            yield str(content)
                    except Exception:
                        continue
        except Exception as e:
            error_msg = f"Error in LangChain agent: {str(e)}"
            logger.error(error_msg)
            yield error_msg

    def get_tool_info(self) -> dict:
        """Get information about available tools."""
        return {
            "mcp_tools": len([t for t in self.tools if isinstance(t, MCPToolWrapper)]),
            "external_tools": len(
                [t for t in self.tools if not isinstance(t, MCPToolWrapper)]
            ),
            "total_tools": len(self.tools),
            "tool_names": [t.name for t in self.tools],
        }

    def _flatten_events(self, event):
        if isinstance(event, dict):
            yield event
        elif isinstance(event, list):
            for item in event:
                yield from self._flatten_events(item)


def _extract_content(ev):
    # Handle dicts
    if isinstance(ev, dict):
        data = ev.get("data", {})
        # If 'chunk' is present, it could be a dict or a custom object
        chunk = data.get("chunk")
        if isinstance(chunk, str):
            return chunk
        elif hasattr(chunk, "content"):
            return getattr(chunk, "content", "")
        # Sometimes the output is in 'output'
        output = data.get("output")
        if isinstance(output, str):
            return output
        elif hasattr(output, "content"):
            return getattr(output, "content", "")
        # Sometimes the output is in 'return_values'
        if "return_values" in data:
            output = data["return_values"].get("output")
            if output:
                return output
    # Handle AIMessageChunk and AgentFinish directly
    elif hasattr(ev, "content"):
        return getattr(ev, "content", "")
    elif hasattr(ev, "return_values"):
        return ev.return_values.get("output", "")
    return None
