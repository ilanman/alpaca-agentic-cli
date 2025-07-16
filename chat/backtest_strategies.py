import pandas as pd
import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import yfinance as yf  # type: ignore
import warnings


class BaseStrategy(ABC):
    """Abstract base class for all trading strategies."""

    @abstractmethod
    def generate_signals(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        """Generate trading signals for the given dataframe."""
        pass

    @abstractmethod
    def get_default_params(self) -> Dict[str, Any]:
        """Return default parameters for the strategy."""
        pass

    @abstractmethod
    def get_description(self) -> str:
        """Return a description of the strategy."""
        pass

    @abstractmethod
    def get_required_params(self) -> Dict[str, str]:
        """Return required parameter names and their types as strings."""
        pass

    @abstractmethod
    def get_ordered_param_prompts(self) -> list:
        """Return a list of dicts: [{name, type, example}] in order for prompting."""
        pass


class SMAStrategy(BaseStrategy):
    """Simple Moving Average Crossover Strategy."""

    def get_default_params(self) -> Dict[str, Any]:
        return {"short_window": 20, "long_window": 50}

    def get_description(self) -> str:
        return "Simple Moving Average Crossover - Buy when short SMA crosses above long SMA"

    def get_required_params(self) -> Dict[str, str]:
        return {
            "symbol": "str (e.g., 'AAPL')",
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
            "short_window": "int",
            "long_window": "int",
        }

    def get_ordered_param_prompts(self) -> list:
        return [
            {"name": "symbol", "type": "str", "example": "AAPL"},
            {"name": "start", "type": "YYYY-MM-DD", "example": "2020-01-01"},
            {"name": "end", "type": "YYYY-MM-DD", "example": "2021-01-01"},
            {"name": "short_window", "type": "int", "example": 20},
            {"name": "long_window", "type": "int", "example": 50},
        ]

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        short_window = params.get("short_window", 20)
        long_window = params.get("long_window", 50)

        df = df.copy()
        df["SMA_short"] = df["Close"].rolling(window=short_window).mean()
        df["SMA_long"] = df["Close"].rolling(window=long_window).mean()
        df = df.dropna()

        # Generate signals using numpy where to avoid Series boolean ambiguity
        # Convert to numpy arrays to ensure no pandas Series boolean operations
        sma_short: np.ndarray = df["SMA_short"].values
        sma_long: np.ndarray = df["SMA_long"].values

        df["signal"] = np.where(
            sma_short > sma_long,
            1,
            np.where(sma_short < sma_long, -1, 0),
        )
        df["position"] = df["signal"].diff()

        return df


class RSIStrategy(BaseStrategy):
    """Relative Strength Index Strategy."""

    def get_default_params(self) -> Dict[str, Any]:
        return {"rsi_period": 14, "overbought": 70, "oversold": 30}

    def get_description(self) -> str:
        return "RSI Strategy - Buy when RSI crosses below oversold, sell when above overbought"

    def get_required_params(self) -> Dict[str, str]:
        return {
            "symbol": "str (e.g., 'AAPL')",
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
            "rsi_period": "int",
            "overbought": "int",
            "oversold": "int",
        }

    def get_ordered_param_prompts(self) -> list:
        return [
            {"name": "symbol", "type": "str", "example": "AAPL"},
            {"name": "start", "type": "YYYY-MM-DD", "example": "2020-01-01"},
            {"name": "end", "type": "YYYY-MM-DD", "example": "2021-01-01"},
            {"name": "rsi_period", "type": "int", "example": 14},
            {"name": "overbought", "type": "int", "example": 70},
            {"name": "oversold", "type": "int", "example": 30},
        ]

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.DataFrame:  # type: ignore
        rsi_period = params.get("rsi_period", 14)
        overbought = params.get("overbought", 70)
        oversold = params.get("oversold", 30)

        df = df.copy()

        # Calculate RSI
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=rsi_period).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))

        df = df.dropna()

        # Generate signals using numpy where to avoid Series boolean ambiguity
        # Convert to numpy arrays to ensure no pandas Series boolean operations
        rsi_values: np.ndarray = df["RSI"].values

        cond1 = rsi_values < oversold
        cond2 = rsi_values > overbought
        df["signal"] = np.where(cond1, 1, np.where(cond2, -1, 0))

        df["position"] = df["signal"].diff()

        return df


class BuyAndHoldStrategy(BaseStrategy):
    """Buy and Hold Strategy - Buy once and hold."""

    def get_default_params(self) -> Dict[str, Any]:
        return {}

    def get_description(self) -> str:
        return "Buy and Hold - Buy at the beginning and hold until the end"

    def get_required_params(self) -> Dict[str, str]:
        return {
            "symbol": "str (e.g., 'AAPL')",
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
        }

    def get_ordered_param_prompts(self) -> list:
        return [
            {"name": "symbol", "type": "str", "example": "AAPL"},
            {"name": "start", "type": "YYYY-MM-DD", "example": "2020-01-01"},
            {"name": "end", "type": "YYYY-MM-DD", "example": "2021-01-01"},
        ]

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        df = df.copy()
        df["signal"] = 0
        df.iloc[0, df.columns.get_loc("signal")] = 1  # type: ignore[index]
        df.iloc[-1, df.columns.get_loc("signal")] = -1  # type: ignore[index]
        df["position"] = df["signal"].diff()

        return df


class BollingerBandsStrategy(BaseStrategy):
    """Bollinger Bands Strategy."""

    def get_default_params(self) -> Dict[str, Any]:
        return {"window": 20, "num_std": 2}

    def get_description(self) -> str:
        return "Bollinger Bands - Buy when price touches lower band, sell when touches upper band"

    def get_required_params(self) -> Dict[str, str]:
        return {
            "symbol": "str (e.g., 'AAPL')",
            "start": "YYYY-MM-DD",
            "end": "YYYY-MM-DD",
            "window": "int",
            "num_std": "int or float",
        }

    def get_ordered_param_prompts(self) -> list:
        return [
            {"name": "symbol", "type": "str", "example": "AAPL"},
            {"name": "start", "type": "YYYY-MM-DD", "example": "2020-01-01"},
            {"name": "end", "type": "YYYY-MM-DD", "example": "2021-01-01"},
            {"name": "window", "type": "int", "example": 20},
            {"name": "num_std", "type": "int or float", "example": 2},
        ]

    def generate_signals(self, df: pd.DataFrame, **params) -> pd.DataFrame:
        window = params.get("window", 20)
        num_std = params.get("num_std", 2)

        df = df.copy()

        # Calculate Bollinger Bands with explicit type handling
        df["BB_middle"] = df["Close"].rolling(window=window).mean()
        bb_std = df["Close"].rolling(window=window).std()

        # Ensure bb_std is a Series (not DataFrame)
        if isinstance(bb_std, pd.DataFrame):
            bb_std = bb_std.iloc[:, 0]

        df["BB_upper"] = df["BB_middle"] + (bb_std * float(num_std))
        df["BB_lower"] = df["BB_middle"] - (bb_std * float(num_std))

        df = df.dropna()

        # Generate signals using numpy where to avoid Series boolean ambiguity
        # Convert to numpy arrays to ensure no pandas Series boolean operations
        close_values: np.ndarray = df["Close"].values
        bb_upper_values: np.ndarray = df["BB_upper"].values
        bb_lower_values: np.ndarray = df["BB_lower"].values

        df["signal"] = np.where(
            close_values <= bb_lower_values,
            1,
            np.where(close_values >= bb_upper_values, -1, 0),
        )

        df["position"] = df["signal"].diff()

        return df


# Strategy Registry
STRATEGY_REGISTRY = {
    "sma_crossover": SMAStrategy(),
    "rsi": RSIStrategy(),
    "buy_and_hold": BuyAndHoldStrategy(),
    "bollinger_bands": BollingerBandsStrategy(),
}


def get_available_strategies() -> Dict[str, str]:
    """Return a dictionary of available strategies and their descriptions."""
    return {
        name: strategy.get_description() for name, strategy in STRATEGY_REGISTRY.items()
    }


def run_strategy_backtest(
    symbol: str, start: str, end: str, strategy_name: str, **strategy_params
) -> Dict[str, Any]:
    """
    Run a backtest using the specified strategy.

    Args:
        symbol: Stock symbol (e.g., 'AAPL')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        strategy_name: Name of the strategy to use
        **strategy_params: Parameters for the strategy

    Returns:
        Dictionary with backtest results
    """
    # Validate strategy
    if strategy_name not in STRATEGY_REGISTRY:
        return {
            "error": f"Strategy '{strategy_name}' not found. Available: {list(STRATEGY_REGISTRY.keys())}"
        }

    # Validate symbol format
    if not symbol or not isinstance(symbol, str):
        return {"error": "Symbol must be a non-empty string (e.g., 'AAPL', 'TSLA')"}

    # Validate date format
    try:
        from datetime import datetime

        start_date = datetime.strptime(start, "%Y-%m-%d")
        end_date = datetime.strptime(end, "%Y-%m-%d")

        if start_date >= end_date:
            return {"error": "Start date must be before end date"}

        if start_date > datetime.now():
            return {"error": "Start date cannot be in the future"}

    except ValueError:
        return {"error": "Invalid date format. Use YYYY-MM-DD (e.g., '2023-01-01')"}

    # Try to download data with better error handling
    try:
        df = yf.download(symbol, start=start, end=end, progress=False, auto_adjust=True)
    except Exception as e:
        return {"error": f"Error downloading data for {symbol}: {str(e)}"}

    if df.empty:
        # Try to get basic info to see if symbol exists
        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            if not info or info.get("regularMarketPrice") is None:
                return {
                    "error": f"Symbol '{symbol}' not found. Please check the symbol and try again."
                }
            else:
                return {
                    "error": f"No historical data available for {symbol} in range {start} to {end}. Try a different date range."
                }
        except Exception:
            return {
                "error": f"Symbol '{symbol}' not found or invalid. Please check the symbol and try again."
            }

    # Get strategy and default parameters
    strategy = STRATEGY_REGISTRY[strategy_name]
    default_params = strategy.get_default_params()
    required_params = strategy.get_required_params()

    # Merge default params with provided params
    final_params = {**default_params, **strategy_params}

    # Strict parameter validation
    missing = []
    wrong_type = []
    for param, typ in required_params.items():
        val = locals().get(param, strategy_params.get(param))
        if val is None:
            missing.append(f"{param} ({typ})")
        else:
            # Type check (basic)
            if "int" in typ and not isinstance(val, int):
                try:
                    int(val)
                except Exception:
                    wrong_type.append(f"{param} should be int, got {val}")
            if "float" in typ and not isinstance(val, (float, int)):
                try:
                    float(val)
                except Exception:
                    wrong_type.append(f"{param} should be float, got {val}")
            if "str" in typ and not isinstance(val, str):
                wrong_type.append(f"{param} should be str, got {val}")
            if "YYYY-MM-DD" in typ and (not isinstance(val, str) or len(val) != 10):
                wrong_type.append(f"{param} should be YYYY-MM-DD, got {val}")

    if missing or wrong_type:
        example = {k: v for k, v in required_params.items()}
        return {
            "error": "Missing or invalid parameters.",
            "missing": missing,
            "wrong_type": wrong_type,
            "required_format": required_params,
            "example": example,
        }

    # Generate signals
    df_with_signals = strategy.generate_signals(df, **final_params)

    # Extract trades
    trades = []
    entry = None
    entry_date = None

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=FutureWarning)
        for idx, row in df_with_signals.iterrows():
            # Convert to scalar values to avoid Series boolean ambiguity
            position = float(row["position"])
            close_price = float(row["Close"])

            if position == 1:  # Buy signal
                entry = close_price
                entry_date = idx.strftime("%Y-%m-%d")  # type: ignore[attr-defined]
            elif position == -1 and entry is not None:  # Sell signal
                exit_price = close_price
                exit_date = idx.strftime("%Y-%m-%d")  # type: ignore[attr-defined]
                pnl = exit_price - entry
                trades.append(
                    {
                        "entry_date": entry_date,
                        "exit_date": exit_date,
                        "entry_price": entry,
                        "exit_price": exit_price,
                        "pnl": pnl,
                        "return_pct": (pnl / entry) * 100,
                    }
                )
                entry = None
                entry_date = None

    # Calculate statistics
    total_pnl = sum(t["pnl"] for t in trades)
    num_trades = len(trades)
    win_trades = len([t for t in trades if t["pnl"] > 0])
    win_rate = win_trades / num_trades if num_trades > 0 else 0

    # Calculate buy and hold return for comparison
    buy_hold_return = (
        (df["Close"].iloc[-1] - df["Close"].iloc[0]) / df["Close"].iloc[0]
    ) * 100

    return {
        "symbol": symbol,
        "start": start,
        "end": end,
        "strategy": strategy_name,
        "strategy_params": final_params,
        "total_pnl": total_pnl,
        "total_return_pct": (total_pnl / df["Close"].iloc[0]) * 100,
        "num_trades": num_trades,
        "win_rate": win_rate,
        "buy_hold_return": buy_hold_return,
        "trades": trades,
        "strategy_description": strategy.get_description(),
    }


def compare_strategies_backtest(
    symbol: str,
    start: str,
    end: str,
    strategy_names: list,
    strategy_params: Optional[dict] = None,
) -> dict:
    """
    Run backtests for multiple strategies separately and return their results for comparison.

    Args:
        symbol: Stock symbol (e.g., 'GS')
        start: Start date (YYYY-MM-DD)
        end: End date (YYYY-MM-DD)
        strategy_names: List of strategy names to run (e.g., ['rsi', 'bollinger_bands'])
        strategy_params: Optional dict mapping strategy_name to its params dict

    Returns:
        Dictionary mapping strategy_name to its backtest result dict
    """
    if strategy_params is None:
        strategy_params = {}

    results = {}
    for strategy_name in strategy_names:
        params = strategy_params.get(strategy_name, {})
        result = run_strategy_backtest(symbol, start, end, strategy_name, **params)
        results[strategy_name] = result

    return results
