# finAI Trading Agent

A recent fix was applied to resolve import errors caused by project restructuring. The project structure was reverted to its original layout to restore module paths and ensure stable agent initialization.

## 🚀 Features

- **🤖 Agent Framework**: Modern agent framework (class name: `Agent`) for trading and research
- **⚡ Powered by LangChain**: Uses LangChain under the hood for LLM orchestration, tool calling, and memory
- **📊 Multiple Data Sources**: Alpaca MCP + yfinance + web search + Wikipedia
- **💬 Natural Language Trading**: Chat with your account in plain English
- **🔄 Streaming Responses**: Real-time, incremental output for a clean CLI experience
- **🧠 Conversation Memory**: Context-aware conversations
- **🛠️ Enhanced Tool Calling**: Robust tool selection and error handling
- **📈 Comprehensive Analysis**: Combine trading data with research and news

---

## Prerequisite: Alpaca MCP Server

This agent relies on the [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server) as a backend for all trading operations.

**To use this project, you must:**
1. Clone and install the official [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server).
2. Provide your own Alpaca API keys by following the [server's setup instructions](https://github.com/alpacahq/alpaca-mcp-server#installation).
3. Start the MCP server (see the server README for full instructions).
4. Use this agent to connect (point the `MCP_SERVER_URL` in your `.env` to your running server).

*For server installation, configuration, and API key setup, always refer to the [Alpaca MCP server README](https://github.com/alpacahq/alpaca-mcp-server#installation).*

---

## Requirements

- Python 3.10+
- [alpaca-mcp-server](https://github.com/alpacahq/alpaca-mcp-server) (run separately, not included)
- OpenAI API key
- Alpaca API and Secret keys

---

## Python Dependencies

- [langchain](https://pypi.org/project/langchain/) – Modern LLM framework
- [langchain-openai](https://pypi.org/project/langchain-openai/) – OpenAI integration
- [langchain-community](https://pypi.org/project/langchain-community/) – Community tools
- [openai](https://pypi.org/project/openai/) – Chat/LLM API
- [mcp](https://pypi.org/project/mcp/) – Model Context Protocol Python SDK
- [python-dotenv](https://pypi.org/project/python-dotenv/) – Load config from .env files
- [yfinance](https://pypi.org/project/yfinance/) – Yahoo Finance data
- [duckduckgo-search](https://pypi.org/project/duckduckgo-search/) – Web search
- [wikipedia](https://pypi.org/project/wikipedia/) – Wikipedia data
- [newsapi-python](https://pypi.org/project/newsapi-python/) – News data

---

## Setup

1. **Clone this repo:**
    ```bash
    git clone https://github.com/your-username/finAI.git
    cd finAI
    ```

2. **Create and activate a virtual environment:**
    ```bash
    python -m venv venv
    source venv/bin/activate
    ```

3. **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4. **Set up your `.env` file:**
    ```bash
    cp .env.example .env
    ```
    Fill in your `OPENAI_API_KEY` and `MCP_SERVER_URL` (e.g. `http://localhost:8000`).

5. **Start the Alpaca MCP server in a separate terminal:**
    ```bash
    # Use the provided script (recommended)
    python start_alpaca_server.py
    
    # OR manually (if you prefer)
    cd alpaca-mcp-server
    python alpaca_mcp_server.py
    ```
    
    *Note: The `alpaca-mcp-server` directory should already exist in this project. If it doesn't, you'll need to clone it from [Alpaca's repository](https://github.com/alpacahq/alpaca-mcp-server) and set up your API keys according to their instructions.*

6. **Run the agent:**
    ```bash
    python main.py
    ```

---

## Example Usage

After setup, start the agent:

```bash
python main.py
```

You will see the agent initialize with available tools. Try these examples:

### Basic Trading
```
> What's my current account balance?
> Get the current price of AAPL
> Buy 1 share of TSLA
> Show me my current positions
```

### Research & Analysis
```
> Search for recent news about Tesla
> What is Tesla's P/E ratio?
> Tell me about Apple's business model
> Get technical indicators for AAPL
```

### Comprehensive Analysis
```
> Analyze AAPL comprehensively: get current price, recent news, and P/E ratio
> Compare Tesla and Apple's financial metrics
> What's the market sentiment around AI stocks?
```

### Testing
Run the test suite to verify functionality:
```bash
python main.py --test
```

---

## Available Tools

### Trading Tools (MCP)
- Account information and balances
- Real-time stock quotes and market data
- Order placement and management
- Position tracking and management
- Options trading
- Watchlist management

### Research Tools (External)
- **Web Search**: DuckDuckGo for real-time information
- **Financial Data**: yfinance for P/E ratios, earnings, dividends
- **Company Info**: Wikipedia for business models and background
- **News**: NewsAPI for market sentiment and developments

---
## Streamlit App Usage

This project includes a Streamlit-based trading chatbot interface for interactive use.

### Running the Streamlit App

1. Ensure you have installed the dependencies as described above.
2. Set up your `.env` file with the required environment variables (`OPENAI_API_KEY`, `MCP_SERVER_URL`).
3. Start the Alpaca MCP server in a separate terminal (see instructions above).
4. Run the Streamlit app with:

```bash
streamlit run streamlit_app.py
```

or if you prefer to use the app directory version:

```bash
streamlit run app/streamlit_app.py
```

### Features

- Interactive chat interface powered by the finAI agent.
- Model selection from supported OpenAI models.
- Sidebar with available backtesting strategies, including detailed descriptions and Investopedia links.
- Real-time streaming responses and conversation history.
- Clear chat button to reset the conversation.

### Notes

- The Streamlit app uses asynchronous calls to the agent for responsive interaction.
- Make sure the MCP server is running and accessible via the `MCP_SERVER_URL` environment variable.
- The app requires Python 3.10+ and the dependencies listed in the requirements.

---
