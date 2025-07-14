import pytest
from chat.agent import MCPToolWrapper
from chat.mcp_client import AlpacaMCPClient


@pytest.fixture
def dummy_mcp_client(monkeypatch):
    client = AlpacaMCPClient("http://dummy")

    async def fake_call_tool(tool_name, kwargs):
        class Result:
            content = "dummy result"

        return Result()

    monkeypatch.setattr(client, "call_tool", fake_call_tool)
    return client


@pytest.mark.asyncio
async def test_validate_stock_order_params_missing_symbol(dummy_mcp_client):
    wrapper = MCPToolWrapper(dummy_mcp_client, "place_stock_order", "desc")
    result = wrapper._validate_stock_order_params({"side": "buy", "quantity": 1})
    assert "Missing required parameters" in result


@pytest.mark.asyncio
async def test_validate_stock_order_params_invalid_quantity(dummy_mcp_client):
    wrapper = MCPToolWrapper(dummy_mcp_client, "place_stock_order", "desc")
    result = wrapper._validate_stock_order_params(
        {"symbol": "AAPL", "side": "buy", "quantity": "one"}
    )
    assert "Quantity must be a number" in result


@pytest.mark.asyncio
async def test_validate_stock_order_params_invalid_side(dummy_mcp_client):
    wrapper = MCPToolWrapper(dummy_mcp_client, "place_stock_order", "desc")
    result = wrapper._validate_stock_order_params(
        {"symbol": "AAPL", "side": "hold", "quantity": 1}
    )
    assert "Side must be 'buy' or 'sell'" in result


@pytest.mark.asyncio
async def test_confirm_trade_yes(dummy_mcp_client, monkeypatch):
    wrapper = MCPToolWrapper(dummy_mcp_client, "place_stock_order", "desc")
    # Mock input to return 'yes'
    monkeypatch.setattr("builtins.input", lambda: "yes")
    # Mock yfinance
    import yfinance

    monkeypatch.setattr(
        yfinance,
        "Ticker",
        lambda symbol: type("T", (), {"info": {"currentPrice": 10}})(),
    )
    confirmed = await wrapper._confirm_trade(
        {"symbol": "AAPL", "side": "buy", "quantity": 1}
    )
    assert confirmed is True


@pytest.mark.asyncio
async def test_confirm_trade_no(dummy_mcp_client, monkeypatch):
    wrapper = MCPToolWrapper(dummy_mcp_client, "place_stock_order", "desc")
    # Mock input to return 'no'
    monkeypatch.setattr("builtins.input", lambda: "no")
    # Mock yfinance
    import yfinance

    monkeypatch.setattr(
        yfinance,
        "Ticker",
        lambda symbol: type("T", (), {"info": {"currentPrice": 10}})(),
    )
    confirmed = await wrapper._confirm_trade(
        {"symbol": "AAPL", "side": "buy", "quantity": 1}
    )
    assert confirmed is False
