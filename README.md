# Alpaca Agentic Trading CLI

A command-line agent that lets you interact with the [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server) using OpenAI-powered LLMs for true natural language, tool-augmented trading. Features include privacy-preserving disposable queries, per-request and cumulative token usage reporting, and robust tool logging.


---

## Prerequisite: Alpaca MCP Server

This agentic CLI relies on the [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server) as a backend for all trading operations.

**To use this project, you must:**
1. Clone and install the official [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server).
2. Provide your own Alpaca API keys by following the [server's setup instructions](https://github.com/alpacahq/alpaca-mcp-server#installation).
3. Start the MCP server (see the server README for full instructions).
4. Use this CLI to connect (point the `MCP_SERVER_URL` in your `.env` to your running server).

*For server installation, configuration, and API key setup, always refer to the [Alpaca MCP server README](https://github.com/alpacahq/alpaca-mcp-server#installation).*

---

## Requirements

- Python 3.10+
- [alpaca-mcp-server](https://github.com/alpacahq/alpaca-mcp-server) (run separately, not included)
- OpenAI API key
- Alpaca API and Secret keys


---

## Python Dependencies

- [openai](https://pypi.org/project/openai/) – Chat/LLM API
- [mcp](https://pypi.org/project/mcp/) – Model Context Protocol Python SDK
- [python-dotenv](https://pypi.org/project/python-dotenv/) – Load config from .env files
- [tiktoken](https://pypi.org/project/tiktoken/) – For token counting in prompt window
- [pytest](https://pypi.org/project/pytest/) – For running tests

---

## Setup

1. **Clone this repo:**
    ```bash
    git clone https://github.com/your-username/alpaca-agentic-cli.git
    cd alpaca-agentic-cli
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

5. **Install and start the [Alpaca MCP server](https://github.com/alpacahq/alpaca-mcp-server) in a separate terminal:**
    ```bash
    git clone https://github.com/alpacahq/alpaca-mcp-server.git
    cd alpaca-mcp-server
    pip install -e .
    python alpaca_mcp_server.py
    ```

6. **Run the CLI:**
    ```bash
    python main.py
    ```

---

## Features & Customizations

- **Natural language trading:** Chat with your account and request trades in plain English.
- **OpenAI tool calling:** Leverages LLM-native function calling for robust, schema-driven trading commands.
- **Disposable mode:** Prefix messages with `-d` to prevent history retention and reduce context cost.
- **Per-request and cumulative token usage reporting:** See OpenAI's official token count for each exchange and the running total for your session, both in the terminal and in `chat_agent.log`.
- **Response source display:** The CLI clearly indicates whether a response was generated directly by the LLM or with tool assistance.
- **Tool call logging:** All MCP tool calls and arguments are logged for full auditability.
- **Error-handled tool loop:** Compliant with OpenAI tool call protocol (no 400 errors).
- **Session memory:** The LLM maintains context by including the full conversation history in each API call.

---

## Example Usage

After setup, start the CLI:

```bash
python main.py
```

You will see a prompt. Try the following:

```
> What's the current price of AAPL?

[TOKEN USAGE]
Current request (including full context history):
  Prompt tokens: 45
  Completion tokens: 20
  Total tokens: 65

Session totals (sum of all API calls):
  Prompt tokens: 45
  Completion tokens: 20
  Total tokens: 65

[RESPONSE SOURCE] Generated using LLM with tool assistance
AAPL is currently trading at $172.34.
```

You can also place trades:

```
> Buy 1 share of TSLA

[TOKEN USAGE]
Current request (including full context history):
  Prompt tokens: 90
  Completion tokens: 30
  Total tokens: 120

Session totals (sum of all API calls):
  Prompt tokens: 135
  Completion tokens: 50
  Total tokens: 185

[RESPONSE SOURCE] Generated using LLM with tool assistance
Order placed: Bought 1 share of TSLA at $700.00.
```

To use disposable mode (no history retained for that turn):

```
> -d What is my account balance?
```

---

## How This Differs from Stock MCP

- Does not modify MCP server code.
- All customizations are in the CLI agent and OpenAI integration.
- Disposable/persistent chat history management for privacy and cost control.
- Token and tool call logging for transparency.

---

## Contributing

PRs and forks welcome!  
See `chat/agent.py` for how to add features or modify disposable logic.

---

## License

MIT
