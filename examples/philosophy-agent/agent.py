"""
Philosophy Agent - A conversational AI agent specializing in philosophical discourse.

This agent uses Anthropic's Claude model with Composio tools to engage in deep
philosophical conversations, look up references, and explore ideas across major
philosophical traditions.

Required environment variables:
- ANTHROPIC_API_KEY: Your Anthropic API key (https://console.anthropic.com)
- COMPOSIO_API_KEY: Your Composio API key (https://app.composio.dev)
"""

import os
from typing import Optional

import anthropic
from composio_anthropic import AnthropicProvider
from dotenv import load_dotenv

from composio import Composio

load_dotenv()

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

MODEL = "claude-3-5-sonnet-20241022"
MAX_TOKENS = 4096
MAX_ITERATIONS = 10  # safeguard against runaway agentic loops

SYSTEM_PROMPT = """You are Sophia, a philosophical AI agent with deep knowledge of Western
and Eastern philosophical traditions. You engage thoughtfully with questions spanning:

- Ancient Greek philosophy (Plato, Aristotle, Stoics, Epicureans)
- Medieval and Renaissance philosophy (Aquinas, Descartes)
- Modern philosophy (Kant, Hegel, Nietzsche, Wittgenstein)
- Contemporary philosophy (analytic, continental, pragmatism)
- Eastern traditions (Buddhism, Taoism, Confucianism, Vedanta)
- Ethics, aesthetics, epistemology, metaphysics, political philosophy

You use the Socratic method — asking probing questions to help the user clarify
and deepen their thinking. When relevant, use your tools to look up philosophical
texts, search for current academic discussions, or retrieve factual context.

Always reason carefully, acknowledge uncertainty, and present multiple perspectives
before offering your own considered view. Cite philosophers and works by name."""

# ---------------------------------------------------------------------------
# Agent setup
# ---------------------------------------------------------------------------

anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

_composio_api_key = os.getenv("COMPOSIO_API_KEY")
composio: Optional[Composio] = None
if _composio_api_key:
    composio = Composio(
        api_key=_composio_api_key,
        provider=AnthropicProvider(),
    )

# Fetch tools useful for a philosophy agent (web search, Wikipedia lookups, etc.)
# Falls back to an empty list if no tools are available for the user.
def get_tools() -> list:
    if not composio:
        return []
    try:
        tools = composio.tools.get(
            user_id="default",
            search="search wikipedia",
        )
        return tools if tools else []
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] Could not fetch Composio tools: {exc}")
        return []


# ---------------------------------------------------------------------------
# Agentic loop
# ---------------------------------------------------------------------------

def run_agent(user_message: str, tools: list) -> str:
    """Run a single agentic turn and return the final text response."""
    messages: list[dict] = [{"role": "user", "content": user_message}]

    for iteration in range(MAX_ITERATIONS):
        response = anthropic_client.messages.create(
            model=MODEL,
            max_tokens=MAX_TOKENS,
            system=SYSTEM_PROMPT,
            tools=tools if tools else [],
            messages=messages,
        )

        # Append assistant turn
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            # Collect all text blocks
            text_parts = [
                block.text for block in response.content if block.type == "text"
            ]
            return "\n".join(text_parts)

        if response.stop_reason == "tool_use" and tools and composio:
            # Execute tool calls via Composio and append results
            tool_results = composio.provider.handle_tool_calls(
                user_id="default", response=response
            )
            messages.extend(tool_results)
        else:
            # Unexpected stop reason or no tools — return whatever text we have
            text_parts = [
                block.text for block in response.content if block.type == "text"
            ]
            return "\n".join(text_parts) if text_parts else "(no response)"

    return "(max iterations reached)"


# ---------------------------------------------------------------------------
# Interactive REPL
# ---------------------------------------------------------------------------

def print_banner() -> None:
    print("=" * 60)
    print("  Sophia — Philosophy Agent")
    print("  Powered by Anthropic Claude + Composio")
    print("=" * 60)
    print("  Ask any philosophical question. Type 'exit' to quit.")
    print("  Examples:")
    print("    > What is the meaning of life?")
    print("    > Explain Kant's categorical imperative.")
    print("    > Compare Stoic and Buddhist views on suffering.")
    print("=" * 60)
    print()


def main(initial_question: Optional[str] = None) -> None:
    print_banner()
    tools = get_tools()

    if tools:
        print(f"[info] Loaded {len(tools)} Composio tool(s).\n")
    else:
        print("[info] Running without Composio tools (pure language model mode).\n")

    if initial_question:
        # Non-interactive: answer a single question and exit
        print(f"Question: {initial_question}\n")
        answer = run_agent(initial_question, tools)
        print(f"Sophia: {answer}\n")
        return

    # Interactive loop
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nFarewell. May your inquiries lead to wisdom.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"exit", "quit", "q"}:
            print("Farewell. May your inquiries lead to wisdom.")
            break

        print()
        answer = run_agent(user_input, tools)
        print(f"Sophia: {answer}\n")


if __name__ == "__main__":
    import sys

    # Optional: pass a question as a CLI argument for non-interactive use
    # e.g.  python agent.py "What is the Socratic method?"
    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else None
    main(initial_question=question)
