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
try:
    from composio_anthropic import AnthropicProvider
    from composio import Composio
    COMPOSIO_AVAILABLE = True
except ImportError:
    COMPOSIO_AVAILABLE = False

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

def ask_sophia(anthropic_client, tools, composio, messages: list) -> str:
    """Send messages to Sophia and return her final response."""
    create_kwargs: dict = {
        "model": "claude-opus-4-5",
        "max_tokens": 4096,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    if tools:
        create_kwargs["tools"] = tools

    while True:
        response = anthropic_client.messages.create(**create_kwargs)
        messages.append({"role": "assistant", "content": response.content})

        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]

        if tool_use_blocks and composio:
            print(f"  [Araştırıyor — {len(tool_use_blocks)} araç kullanılıyor...]")
            tool_results = composio.provider.handle_tool_calls(
                user_id="default", response=response
            )
            messages.extend(tool_results)
            create_kwargs["messages"] = messages
        else:
            return "\n".join(b.text for b in response.content if b.type == "text")


def main() -> None:
    anthropic_client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

    tools = []
    composio = None

    if COMPOSIO_AVAILABLE:
        try:
            composio = Composio(
                api_key=os.environ["COMPOSIO_API_KEY"],
                provider=AnthropicProvider(),
            )
            print("Fetching research tools from Composio...")
            tools = composio.tools.get(user_id="default", toolkits=["TAVILY"])
            print(f"Fetched {len(tools)} tool(s)")
        except Exception as e:
            print(f"Composio unavailable ({type(e).__name__}), running without research tools.")
            tools = []
            composio = None

    print("\n" + "=" * 50)
    print("  Sophia — Felsefe Ajanı")
    print("=" * 50)
    print("Çıkmak için 'quit' veya 'exit' yazın.\n")

    messages: list = []

    while True:
        try:
            user_input = input("Sen: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nHoşça kal!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "çıkış", "çık"):
            print("Hoşça kal!")
            break

        messages.append({"role": "user", "content": user_input})
        print("\nSophia: ", end="", flush=True)
        reply = ask_sophia(anthropic_client, tools, composio, messages)
        print(reply)
        print()


if __name__ == "__main__":
    main()
