import streamlit as st
import asyncio
import os
from dotenv import load_dotenv

from chat.mcp_client import AlpacaMCPClient
from chat.agent import Agent
from chat.backtest_strategies import get_available_strategies

load_dotenv()

# Detailed Investopedia-style descriptions and links for each strategy
investopedia_links = {
    "sma_crossover": "https://www.investopedia.com/terms/m/movingaverage.asp",
    "rsi": "https://www.investopedia.com/terms/r/rsi.asp",
    "buy_and_hold": "https://www.investopedia.com/terms/b/buyandhold.asp",
    "bollinger_bands": "https://www.investopedia.com/terms/b/bollingerbands.asp",
}
investopedia_paragraphs = {
    "sma_crossover": (
        "The Simple Moving Average (SMA) Crossover strategy is a popular trend-following approach that uses two moving averages of different lengths. "
        "A buy signal is generated when the short-term SMA crosses above the long-term SMA, indicating upward momentum, while a sell signal occurs when the short-term SMA crosses below the long-term SMA. "
        "This method helps traders identify potential trend reversals and filter out market noise. "
        "It is most effective in trending markets but can produce false signals during sideways or choppy conditions. "
        "<a href='https://www.investopedia.com/terms/m/movingaverage.asp' target='_blank'>Learn more on Investopedia</a>."
    ),
    "rsi": (
        "The Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements, typically over a 14-day period. "
        "It ranges from 0 to 100 and is used to identify overbought or oversold conditions in a security. "
        "An RSI below 30 is considered oversold (potential buy), while an RSI above 70 is considered overbought (potential sell). "
        "Traders use RSI to spot potential reversals or confirm trends, but it can give misleading signals in strong trending markets. "
        "<a href='https://www.investopedia.com/terms/r/rsi.asp' target='_blank'>Learn more on Investopedia</a>."
    ),
    "buy_and_hold": (
        "The Buy and Hold strategy is a long-term investment approach where an investor purchases a security and holds it for an extended period, regardless of market fluctuations. "
        "This strategy is based on the belief that, in the long run, financial markets tend to rise, and holding investments through volatility will yield positive returns. "
        "It minimizes transaction costs and capital gains taxes, making it popular among passive investors. "
        "However, it does not provide risk management or profit-taking mechanisms. "
        "<a href='https://www.investopedia.com/terms/b/buyandhold.asp' target='_blank'>Learn more on Investopedia</a>."
    ),
    "bollinger_bands": (
        "Bollinger Bands are a technical analysis tool consisting of a middle band (SMA) and two outer bands set a certain number of standard deviations away. "
        "They help traders identify periods of high or low volatility and potential overbought or oversold conditions. "
        "When the price touches the upper band, it may be overbought; when it touches the lower band, it may be oversold. "
        "Bollinger Bands are often used to spot breakouts, reversals, and volatility expansions. "
        "<a href='https://www.investopedia.com/terms/b/bollingerbands.asp' target='_blank'>Learn more on Investopedia</a>."
    ),
}


def get_event_loop():
    """Get or create the asyncio event loop."""
    try:
        return asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop


@st.cache_resource(show_spinner=True)
def create_agent(model=None):
    """Create and cache the agent and client for a given model."""
    server_url = os.getenv("MCP_SERVER_URL", "http://localhost:8000")
    loop = get_event_loop()
    mcp_client = AlpacaMCPClient(server_url)
    agent = loop.run_until_complete(
        Agent.create(mcp_client, model) if model else Agent.create(mcp_client)
    )
    return agent, mcp_client, loop


@st.cache_resource(show_spinner=True)
def get_agent_cache():
    """Cache for agents per model."""
    return {}


def render_sidebar():
    """Render the sidebar with branding, model selection, and available strategies as clickable links with details."""
    st.sidebar.title("finAI Trading Chatbot")
    st.sidebar.markdown(
        """
        This is a trading chatbot powered by finAI.\nEnter your queries below and get responses.
        """
    )
    model_options = [
        "gpt-4o",
        "gpt-4o-mini",
        "gpt-3.5-turbo",
        "gpt-3.5-turbo-16k",
    ]
    selected_model = st.sidebar.selectbox(
        "Select Model", model_options, index=0, label_visibility="visible"
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown(
        '<div style="background:#222; border-radius:10px; padding:12px 14px 8px 14px; margin-bottom:10px;">'
        '<span style="font-size:16px; font-weight:600; color:#fff;">Available Backtesting Strategies:</span><br>',
        unsafe_allow_html=True,
    )
    strategies = get_available_strategies()
    # Hardcoded details for each strategy
    strategy_details = {
        "sma_crossover": {
            "title": "SMA Crossover",
            "pros": [
                "Simple to understand and implement",
                "Works well in trending markets",
            ],
            "cons": ["Lags price action", "Prone to whipsaws in sideways markets"],
            "when": "Use when you expect clear trends and want a simple rule-based approach.",
        },
        "rsi": {
            "title": "RSI",
            "pros": [
                "Good for identifying overbought/oversold conditions",
                "Widely used and trusted",
            ],
            "cons": ["Can give false signals in strong trends", "Parameter sensitive"],
            "when": "Use for mean-reverting assets or to time entries/exits in range-bound markets.",
        },
        "buy_and_hold": {
            "title": "Buy and Hold",
            "pros": [
                "Very simple",
                "Low transaction costs",
                "Captures long-term growth",
            ],
            "cons": ["No risk management", "No profit taking or stop loss"],
            "when": "Use for long-term investing in assets you believe will appreciate.",
        },
        "bollinger_bands": {
            "title": "Bollinger Bands",
            "pros": ["Adapts to volatility", "Can identify breakouts and reversals"],
            "cons": ["Parameter sensitive", "Can be confusing in choppy markets"],
            "when": "Use to spot volatility expansions or mean reversion opportunities.",
        },
    }
    if "selected_strategy" not in st.session_state:
        st.session_state["selected_strategy"] = None
    # Arrange all strategy buttons in a single row, tightly packed
    strategy_names = list(strategies.keys())
    cols = st.sidebar.columns([1] * len(strategy_names))  # Even, tight columns
    for idx, name in enumerate(strategy_names):
        btn_label = f"{strategy_details[name]['title']}"
        if cols[idx].button(btn_label):
            if st.session_state["selected_strategy"] == name:
                st.session_state["selected_strategy"] = (
                    None  # Toggle off if already open
                )
            else:
                st.session_state["selected_strategy"] = name  # Open new or switch

    st.sidebar.markdown("---")

    # Show dialog/expander for selected strategy
    if st.session_state["selected_strategy"]:
        details = strategy_details[st.session_state["selected_strategy"]]
        with st.sidebar.expander("", expanded=True):
            st.markdown(
                f"<div style='font-size:1.5rem; font-weight:700; margin-bottom:0.5em;'>About {details['title']}</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                "<style>ul.tight-list {margin:0 0 0 18px; padding:0;} ul.tight-list li {margin-bottom:2px;}</style>",
                unsafe_allow_html=True,
            )
            # Use the detailed paragraph with Investopedia link
            st.markdown(
                investopedia_paragraphs[st.session_state["selected_strategy"]],
                unsafe_allow_html=True,
            )
            st.markdown("**Pros:**")
            pros_html = (
                '<ul class="tight-list">'
                + "".join(f"<li>{p}</li>" for p in details["pros"])
                + "</ul>"
            )
            st.markdown(pros_html, unsafe_allow_html=True)
            st.markdown("**Cons:**")
            cons_html = (
                '<ul class="tight-list">'
                + "".join(f"<li>{c}</li>" for c in details["cons"])
                + "</ul>"
            )
            st.markdown(cons_html, unsafe_allow_html=True)
            st.markdown(f"**When to use:** {details['when']}")

    if st.sidebar.button("Clear Chat"):
        st.session_state["history"] = []
    return selected_model


def render_chat_history():
    """Display the chat history as a simple stream, with a person at laptop for user and a robot image for FinAI Agent, no background shading."""
    robot_img_url = "https://em-content.zobj.net/source/microsoft-teams/337/robot_1f916.png"  # Microsoft Teams robot emoji PNG
    for user_msg, bot_msg in st.session_state["history"]:
        st.markdown(
            f'<span style="font-size:18px; margin-right:8px;">🧑‍💻</span>'
            f"<b>You:</b> {user_msg}",
            unsafe_allow_html=True,
        )
        st.markdown(
            f'<img src="{robot_img_url}" alt="FinAI Agent" width="20" style="vertical-align:middle; margin-right:8px;">'
            f"<b>FinAI Agent:</b> {bot_msg}",
            unsafe_allow_html=True,
        )


def handle_user_input(agent, loop):
    # Clear the input box before rendering the widget, if flagged
    if st.session_state.get("clear_input", False):
        st.session_state["input_form_input"] = ""
        st.session_state["clear_input"] = False

    # Define a callback to set the submitted flag
    def set_submitted():
        # This function is called when the user presses Enter in the text_input.
        # We set a 'submitted' flag to be handled on the next script rerun.
        st.session_state.submitted = True

    # Render the input box first. It is disabled while the bot is processing.
    st.text_input(
        "Your message",
        key="input_form_input",
        placeholder="Type your message and press Enter",
        on_change=set_submitted,
        disabled=st.session_state.get("processing", False),
    )

    # If a message was submitted, set the 'processing' state and rerun.
    if st.session_state.get("submitted"):
        user_input = st.session_state.input_form_input
        st.session_state.submitted = False  # Consume the flag
        st.session_state.processing = True  # Start processing
        st.session_state.history.append((user_input, "thinking..."))
        st.session_state.clear_input = True  # Set flag to clear input on next run
        st.rerun()

    # On the next rerun, if we are in the 'processing' state, call the agent.
    if st.session_state.get("processing"):
        user_input = st.session_state.history[-1][0]

        with st.spinner("Bot is thinking..."):
            try:
                bot_response = loop.run_until_complete(agent.chat(user_input))
            except Exception as e:
                bot_response = None
                st.error(f"Error: {str(e)}")

        # Update history with the final response.
        st.session_state.history[-1] = (
            user_input,
            bot_response or "[Error: Could not get response]",
        )

        # We are done processing.
        st.session_state.processing = False

        # Rerun to display the final response and re-enable the input box.
        st.rerun()


def main():
    """Main entry point for the Streamlit app."""
    st.set_page_config(
        page_title="finAI Trading Chatbot", page_icon="🤖", layout="wide"
    )
    if "history" not in st.session_state:
        st.session_state["history"] = []
    selected_model = render_sidebar()
    agent_cache = get_agent_cache()
    if selected_model not in agent_cache:
        agent_cache[selected_model] = create_agent(selected_model)
    agent, mcp_client, loop = agent_cache[selected_model]
    # Centered chat interface
    container = st.container()
    with container:
        left_col, center_col, right_col = st.columns([1, 3, 1])
        with center_col:
            render_chat_history()
            handle_user_input(agent, loop)


if __name__ == "__main__":
    main()
