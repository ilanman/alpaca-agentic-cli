"""
finAI Chat Package

This package contains the core chat and agent functionality for the finAI
trading system.

Classes:
    - AlpacaMCPClient: MCP client for communicating with Alpaca
    - LangChainTradingAgent: Enhanced trading agent with LangChain

Author: finAI Team
License: MIT
"""

from .mcp_client import AlpacaMCPClient
from .agent import Agent

__all__ = ["AlpacaMCPClient", "Agent"]
__version__ = "1.0.0"
