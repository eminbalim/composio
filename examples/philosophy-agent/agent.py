"""
Philosophy Agent using Anthropic and Composio SDK

A conversational agent that engages in philosophical discourse and
uses Composio tools to research topics.

Required environment variables:
  COMPOSIO_API_KEY  - Your Composio API key (https://app.composio.dev)
  ANTHROPIC_API_KEY - Your Anthropic API key (https://console.anthropic.com)

Usage:
  python agent.py
"""

import os
from dotenv import load_dotenv
import anthropic
from composio_anthropic import AnthropicProvider
from composio import Composio

load_dotenv()

SYSTEM_PROMPT = """You are a philosophy agent with deep knowledge of philosophical traditions,
thinkers, and ideas spanning from ancient Greece to contemporary philosophy.

Your role is to:
- Engage thoughtfully with philosophical questions and ideas
- Draw connections between different philosophical traditions and thinkers
- Use available tools to research and fact-check philosophical claims
- Present balanced perspectives from multiple philosophical viewpoints
- Help users explore fundamental questions about existence, knowledge, ethics, and meaning

When discussing philosophy, cite relevant thinkers and their works, and explain
complex ideas in accessible language while maintaining intellectual rigor."""

QUESTION = (
    "What is the relationship between Plato's Theory of Forms and modern mathematical "
    "Platonism? How do contemporary philosophers view this connection?"
)


def main() -> None:
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
    composio = Composio(
        api_key=os.environ["COMPOSIO_API_KEY"],
        provider=AnthropicProvider(),
    )

    print("Fetching research tools from Composio...")
    tools = composio.tools.get(user_id="default", toolkits=["TAVILY"])
    print(f"Fetched {len(tools)} tool(s)")

    print("\nPhilosophy Agent")
    print("=" * 50)
    print(f"Question: {QUESTION}")
    print("=" * 50)

    messages = [{"role": "user", "content": QUESTION}]

    # Agentic loop — continue until the model stops calling tools
    while True:
        response = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if tool_use_blocks:
            print(f"\nUsing {len(tool_use_blocks)} research tool(s)...")
            tool_results = composio.provider.handle_tool_calls(
                user_id="default", response=response
            )
            messages.extend(tool_results)
        else:
            # No more tool calls — print the final response
            final_text = "\n".join(
                b.text for b in response.content if b.type == "text"
            )
            print("\nAgent Response:")
            print("-" * 50)
            print(final_text)
            break


if __name__ == "__main__":
    main()
