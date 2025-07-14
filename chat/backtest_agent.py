from typing import Dict, Any
from langchain.tools import BaseTool
from .backtest_strategies import (
    run_strategy_backtest,
    get_available_strategies,
    STRATEGY_REGISTRY,
)


def simple_moving_average_backtest(
    symbol: str,
    start: str,
    end: str,
    short_window: int = 20,
    long_window: int = 50,
) -> Dict[str, Any]:
    """
    Run a simple moving average crossover backtest using yfinance data.
    Returns summary statistics and trades.
    """
    return run_strategy_backtest(
        symbol=symbol,
        start=start,
        end=end,
        strategy_name="sma_crossover",
        short_window=short_window,
        long_window=long_window,
    )


class BacktestTool(BaseTool):
    name: str = "backtest"
    description: str = (
        "Run a backtest using various trading strategies. "
        "You must provide parameters in the exact order and format required by the strategy. "
        "Use 'list_strategies' to see available strategies and their required parameters. "
        "The bot will prompt you for each parameter in order."
    )

    def _run(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        # Support user cancellation
        cancel_keywords = {"cancel", "exit", "stop"}
        for v in input_dict.values():
            if isinstance(v, str) and v.strip().lower() in cancel_keywords:
                return {
                    "message": "Backtest setup cancelled. If you want to start over, just let me know!"
                }
        # Extract strategy_name first
        strategy_name = input_dict.get("strategy_name")
        if not strategy_name:
            strategies = get_available_strategies()
            return {
                "error": "strategy_name is required. Use exact name from available strategies.",
                "available_strategies": strategies,
                "suggestion": "Use 'list_strategies' tool to see all options, then specify exact strategy_name",
            }
        if strategy_name not in STRATEGY_REGISTRY:
            return {
                "error": f"Unknown strategy_name '{strategy_name}'. Use 'list_strategies' to see valid options."
            }
        strategy = STRATEGY_REGISTRY[strategy_name]
        ordered_prompts = strategy.get_ordered_param_prompts()
        # Build current state of params
        params = {k: v for k, v in input_dict.items() if k != "strategy_name"}
        # Find first missing or invalid param
        for prompt in ordered_prompts:
            name = prompt["name"]
            typ = prompt["type"]
            example = prompt["example"]
            val = params.get(name)
            if val is None:
                return {
                    "prompt": f"Please provide '{name}' ({typ}, e.g., '{example}') for strategy '{strategy_name}'.",
                    "template": ordered_prompts,
                }
            # Type validation (basic)
            if typ == "int":
                try:
                    int(val)
                except Exception:
                    return {
                        "prompt": f"'{name}' must be an integer (e.g., {example}). Please provide a valid value.",
                        "template": ordered_prompts,
                    }
            if typ == "float" or typ == "int or float":
                try:
                    float(val)
                except Exception:
                    return {
                        "prompt": f"'{name}' must be a number (e.g., {example}). Please provide a valid value.",
                        "template": ordered_prompts,
                    }
            if typ == "YYYY-MM-DD":
                from datetime import datetime

                try:
                    datetime.strptime(val, "%Y-%m-%d")
                except Exception:
                    return {
                        "prompt": f"'{name}' must be in YYYY-MM-DD format (e.g., {example}). Please provide a valid value.",
                        "template": ordered_prompts,
                    }
            if typ == "str" and not isinstance(val, str):
                return {
                    "prompt": f"'{name}' must be a string (e.g., {example}). Please provide a valid value.",
                    "template": ordered_prompts,
                }
        # All params present and valid, run backtest
        params_with_strategy = dict(params)
        params_with_strategy["strategy_name"] = strategy_name
        return run_strategy_backtest(**params_with_strategy)

    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return self._run(kwargs)


class ListStrategiesTool(BaseTool):
    name: str = "list_strategies"
    description: str = (
        "List all available backtesting strategies with their descriptions."
    )

    def _run(self, input_dict: Dict[str, Any] = {}) -> Dict[str, Any]:
        strategies = get_available_strategies()
        return {"available_strategies": strategies, "count": len(strategies)}

    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return self._run(kwargs)
