from typing import Dict, Any, Tuple, List
from langchain.tools import BaseTool
from .backtest_strategies import (
    run_strategy_backtest,
    get_available_strategies,
    STRATEGY_REGISTRY,
    compare_strategies_backtest,
)


def parse_backtest_input(input_str: str) -> Tuple[Dict[str, Any], List[str]]:
    """Parse a single-line, comma-separated input string into a parameter dict and collect errors."""
    parts = [p.strip() for p in input_str.split(",")]
    result = {}
    errors = []
    for part in parts:
        if "=" in part:
            k, v = part.split("=", 1)
            key = k.strip()
            value = v.strip()
            if not key:
                errors.append(f"Empty key in part '{part}'")
            else:
                result[key] = value
        else:
            errors.append(f"Malformed part without '=': '{part}'")
    return result, errors


class BacktestTool(BaseTool):
    name: str = "backtest"
    description: str = (
        "Run a backtest using various trading strategies. "
        "You must provide all required parameters in a single message, in the exact format required by the strategy. "
        "Use 'list_strategies' to see available strategies and their required parameters. "
        "If you provide incomplete or invalid input, you will receive a detailed error message and an example."
    )

    def _run(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        # If the input is a single string (from a one-line user message), parse it
        if isinstance(input_dict, dict) and len(input_dict) == 1 and None in input_dict:
            # This can happen if the input is passed as {None: '...'}
            input_str = input_dict[None]  # type: ignore
            input_dict, errors = parse_backtest_input(input_str)
        elif isinstance(input_dict, str):
            input_dict, errors = parse_backtest_input(input_dict)
        else:
            errors = []

        # Support user cancellation
        cancel_keywords = {"cancel", "exit", "stop"}
        for v in input_dict.values():
            if isinstance(v, str) and v.strip().lower() in cancel_keywords:
                return {
                    "message": "Backtest setup cancelled. If you want to start over, just let me know!"
                }

        if errors:
            return {
                "error": "Malformed input detected.",
                "details": errors,
                "suggestion": "Please ensure your input is in the format: backtest: key1=value1, key2=value2, ...",
            }

        strategy_name = input_dict.get("strategy_name")
        if not strategy_name:
            strategies = get_available_strategies()
            return {
                "error": "strategy_name is required. Use exact name from available strategies.",
                "available_strategies": strategies,
                "suggestion": "Use 'list_strategies' tool to see all options, then specify exact strategy_name",
            }

        # Check if input_dict contains multiple backtest requests (list of dicts)
        if isinstance(input_dict, list):
            results = []
            for single_input in input_dict:
                res = self.run_single_backtest(single_input)
                results.append(res)
            return {"batch_results": results}

        if strategy_name not in STRATEGY_REGISTRY:
            return {
                "error": f"Unknown strategy_name '{strategy_name}'. Use 'list_strategies' to see valid options."
            }
        strategy = STRATEGY_REGISTRY[strategy_name]
        ordered_prompts = strategy.get_ordered_param_prompts()
        # Always check for missing or wrong-typed params before running
        missing = [p["name"] for p in ordered_prompts if p["name"] not in input_dict]
        wrong_type = []
        for p in ordered_prompts:
            name = p["name"]
            expected_type = p["type"]
            if name in input_dict:
                value = input_dict[name]
                # Only check type if not missing
                if expected_type == "int":
                    try:
                        int(value)
                    except Exception:
                        wrong_type.append(name)
                elif expected_type == "float":
                    try:
                        float(value)
                    except Exception:
                        wrong_type.append(name)
                elif expected_type == "str":
                    if not isinstance(value, str):
                        wrong_type.append(name)
        param_list = "\n".join(
            [
                f"{p['name']}: <{p['type']}> (e.g., {p['example']})"
                for p in ordered_prompts
            ]
        )
        example_lines = [f"{p['name']}={p['example']}" for p in ordered_prompts]
        example_oneline = ", ".join(example_lines)
        example_multiline = "\n".join(
            [f"{p['name']}: {p['example']}" for p in ordered_prompts]
        )
        if missing or wrong_type:
            return {
                "prompt": (
                    f"To use the backtesting agent for the '{strategy_name}' strategy, please copy and paste the following format (all in one line), replacing the values as needed:\n\n"
                    f"backtest: {example_oneline}\n\nOr, in key-value format:\n{param_list}\n\nExample (multi-line):\n{example_multiline}\n\nYou must start your message with 'backtest:' and provide all required parameters in the format above."
                ),
                "required_format": {p["name"]: p["type"] for p in ordered_prompts},
                "example_oneline": example_oneline,
                "example_multiline": example_multiline,
                "missing": missing,
                "wrong_type": wrong_type,
                "error": "Missing or invalid parameters. Please follow the required format exactly.",
            }
        # All required params present and types correct, run the backtest
        # Convert types as needed
        casted_inputs: dict[str, object] = {}
        for p in ordered_prompts:
            name = p["name"]
            expected_type = p["type"]
            value = input_dict[name]
            if expected_type == "int":
                casted_inputs[name] = int(float(value))
            elif expected_type == "float":
                casted_inputs[name] = float(value)
            else:
                casted_inputs[name] = value
        # Extract positional arguments
        symbol = str(casted_inputs.pop("symbol", ""))
        start = str(casted_inputs.pop("start", ""))
        end = str(casted_inputs.pop("end", ""))
        strategy_name_str = str(casted_inputs.pop("strategy_name", strategy_name))
        result = run_strategy_backtest(
            symbol, start, end, strategy_name_str, **casted_inputs
        )
        if "error" in result:
            # If error is about missing/invalid params, show required format and example
            if any(
                k in result
                for k in ("missing", "wrong_type", "required_format", "example")
            ):
                return {
                    "error": result["error"],
                    "missing": result.get("missing", []),
                    "wrong_type": result.get("wrong_type", []),
                    "required_format": result.get("required_format", {}),
                    "example_oneline": example_oneline,
                    "example_multiline": example_multiline,
                    "suggestion": (
                        "Please provide all required fields in the correct format. "
                        f"Example (all in one line):\n{example_oneline}\n"
                        + "Or, multi-line:\n"
                        + example_multiline
                    ),
                }
            else:
                return result
        return result

    def run_single_backtest(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run a single backtest given input parameters dictionary.
        """
        strategy_name = input_dict.get("strategy_name")
        if not strategy_name:
            return {"error": "strategy_name is required for backtest."}
        # Extract known parameters
        symbol = input_dict.get("symbol")
        start = input_dict.get("start")
        end = input_dict.get("end")
        # Remove known keys to pass the rest as strategy_params
        strategy_params = {
            k: v
            for k, v in input_dict.items()
            if k not in {"strategy_name", "symbol", "start", "end"}
        }
        # Ensure symbol, start, end, strategy_name are str or None handled
        symbol_str = str(symbol) if symbol is not None else ""
        start_str = str(start) if start is not None else ""
        end_str = str(end) if end is not None else ""
        strategy_name_str = str(strategy_name) if strategy_name is not None else ""
        return run_strategy_backtest(
            symbol_str, start_str, end_str, strategy_name_str, **strategy_params
        )

        if strategy_name not in STRATEGY_REGISTRY:
            return {
                "error": f"Unknown strategy_name '{strategy_name}'. Use 'list_strategies' to see valid options."
            }
        strategy = STRATEGY_REGISTRY[strategy_name]
        ordered_prompts = strategy.get_ordered_param_prompts()
        # Always check for missing or wrong-typed params before running
        missing = [p["name"] for p in ordered_prompts if p["name"] not in input_dict]
        wrong_type = []
        for p in ordered_prompts:
            name = p["name"]
            expected_type = p["type"]
            if name in input_dict:
                value = input_dict[name]
                # Only check type if not missing
                if expected_type == "int":
                    try:
                        int(value)
                    except Exception:
                        wrong_type.append(name)
                elif expected_type == "float":
                    try:
                        float(value)
                    except Exception:
                        wrong_type.append(name)
                elif expected_type == "str":
                    if not isinstance(value, str):
                        wrong_type.append(name)
        param_list = "\n".join(
            [
                f"{p['name']}: <{p['type']}> (e.g., {p['example']})"
                for p in ordered_prompts
            ]
        )
        example_lines = [f"{p['name']}={p['example']}" for p in ordered_prompts]
        example_oneline = ", ".join(example_lines)
        example_multiline = "\n".join(
            [f"{p['name']}: {p['example']}" for p in ordered_prompts]
        )
        if missing or wrong_type:
            return {
                "prompt": (
                    f"To use the backtesting agent for the '{strategy_name}' strategy, please copy and paste the following format (all in one line), replacing the values as needed:\n\n"
                    f"backtest: {example_oneline}\n\nOr, in key-value format:\n{param_list}\n\nExample (multi-line):\n{example_multiline}\n\nYou must start your message with 'backtest:' and provide all required parameters in the format above."
                ),
                "required_format": {p["name"]: p["type"] for p in ordered_prompts},
                "example_oneline": example_oneline,
                "example_multiline": example_multiline,
                "missing": missing,
                "wrong_type": wrong_type,
                "error": "Missing or invalid parameters. Please follow the required format exactly.",
            }
        # All required params present and types correct, run the backtest
        # Convert types as needed
        casted_inputs = {}
        for p in ordered_prompts:
            name = p["name"]
            expected_type = p["type"]
            value = input_dict[name]
            if expected_type == "int":
                casted_inputs[name] = int(value)
            elif expected_type == "float":
                casted_inputs[name] = float(value)
            else:
                casted_inputs[name] = value
        # Always include strategy_name
        casted_inputs["strategy_name"] = strategy_name
        result = run_strategy_backtest(**casted_inputs)
        if "error" in result:
            # If error is about missing/invalid params, show required format and example
            if any(
                k in result
                for k in ("missing", "wrong_type", "required_format", "example")
            ):
                return {
                    "error": result["error"],
                    "missing": result.get("missing", []),
                    "wrong_type": result.get("wrong_type", []),
                    "required_format": result.get("required_format", {}),
                    "example_oneline": example_oneline,
                    "example_multiline": example_multiline,
                    "suggestion": (
                        "Please provide all required fields in the correct format. "
                        f"Example (all in one line):\n{example_oneline}\n"
                        + "Or, multi-line:\n"
                        + example_multiline
                    ),
                }
            else:
                return result
        return result

    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return self._run(kwargs)


class CompareStrategiesTool(BaseTool):
    name: str = "compare_strategies"
    description: str = (
        "Run backtests for multiple strategies separately and return their results for comparison. "
        "Parameters: symbol (str), start (str), end (str), strategies (list of strategy names), "
        "strategy_params (optional dict mapping strategy name to params dict)."
    )

    def _run(self, input_dict: Dict[str, Any]) -> Dict[str, Any]:
        symbol = input_dict.get("symbol")
        start = input_dict.get("start")
        end = input_dict.get("end")
        strategies = input_dict.get("strategies")
        strategy_params = input_dict.get("strategy_params", {})

        if not symbol or not start or not end or not strategies:
            return {
                "error": "Missing required parameters. Required: symbol, start, end, strategies (list)."
            }

        if not isinstance(strategies, list):
            return {"error": "Parameter 'strategies' must be a list of strategy names."}

        results = compare_strategies_backtest(
            symbol=symbol,
            start=start,
            end=end,
            strategy_names=strategies,
            strategy_params=strategy_params,
        )
        return results

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


class InputInstructionsTool(BaseTool):
    name: str = "input_instructions"
    description: str = (
        "Show detailed input instructions and examples for all available backtesting strategies."
    )

    def _run(self, input_dict: Dict[str, Any] = {}) -> Dict[str, Any]:
        instructions = []
        for name, strategy in STRATEGY_REGISTRY.items():
            ordered_prompts = strategy.get_ordered_param_prompts()
            example_lines = [f"{p['name']}={p['example']}" for p in ordered_prompts]
            example_oneline = ", ".join(example_lines)
            param_list = "\n".join(
                [
                    f"- {p['name']}: <{p['type']}> (e.g., {p['example']})"
                    for p in ordered_prompts
                ]
            )
            instructions.append(
                f"Strategy: {name}\nDescription: {strategy.get_description()}\nRequired parameters:\n{param_list}\nExample (one line):\nbacktest: strategy_name={name}, {example_oneline}\n"
            )
        return {"input_instructions": "\n\n".join(instructions)}

    async def _arun(self, **kwargs) -> Dict[str, Any]:
        return self._run(kwargs)
