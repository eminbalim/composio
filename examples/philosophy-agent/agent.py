"""
Philosophy Agent - An AI agent that explores philosophical questions
using Composio tools and Anthropic Claude.

Required environment variables:
  COMPOSIO_API_KEY  - Your Composio API key (https://app.composio.dev)
  ANTHROPIC_API_KEY - Your Anthropic API key (https://console.anthropic.com)

Usage:
  python agent.py
  python agent.py "What is the meaning of life?"
"""

import os
import sys

import anthropic
from composio_anthropic import AnthropicProvider

from composio import Composio

SYSTEM_PROMPT = """You are a philosophy agent with deep knowledge of philosophical traditions,
thinkers, and ideas from across history and cultures. Your role is to explore philosophical
questions with intellectual rigor, nuance, and curiosity.

When presented with a philosophical question or topic:
1. Draw on relevant philosophical traditions, schools of thought, and key thinkers
2. Use available tools to research current philosophical discourse when helpful
3. Present multiple perspectives and counterarguments
4. Synthesize insights into a coherent, thoughtful response
5. Encourage further exploration and critical thinking

You engage with philosophy not just as an academic exercise but as a genuine inquiry
into the fundamental nature of existence, knowledge, ethics, and human experience."""

DEFAULT_QUESTION = (
    "What is the nature of consciousness and how do different philosophical traditions "
    "approach the mind-body problem? Consider perspectives from Descartes, Eastern philosophy, "
    "and contemporary philosophy of mind."
)


def run_philosophy_agent(question: str = DEFAULT_QUESTION) -> str:
    """Run the philosophy agent with a given question.

    :param question: The philosophical question to explore.
    :return: The agent's response.
    """
    # Initialize clients
    anthropic_client = anthropic.Anthropic()
    composio = Composio(provider=AnthropicProvider())

    # Get tools for web search
    tools = composio.tools.get(user_id="default", toolkits=["TAVILY"])

    print(f"Philosophy Agent\n{'=' * 50}")
    print(f"Question: {question}\n{'=' * 50}\n")

    messages = [{"role": "user", "content": question}]

    # Agentic loop
    while True:
        response = anthropic_client.messages.create(
            model="claude-opus-4-5",
            max_tokens=4096,
            system=SYSTEM_PROMPT,
            tools=tools,
            messages=messages,
        )

        # Add assistant response to messages
        messages.append({"role": "assistant", "content": response.content})

        # Check if we're done (no more tool calls)
        if response.stop_reason == "end_turn":
            for block in response.content:
                if hasattr(block, "text"):
                    return block.text
            return ""

        # Handle tool calls
        tool_results = composio.provider.handle_tool_calls(
            user_id="default", response=response
        )

        # Build tool result messages
        tool_result_content = []
        tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
        for i, result in enumerate(tool_results):
            if i < len(tool_use_blocks):
                tool_result_content.append(
                    {
                        "type": "tool_result",
                        "tool_use_id": tool_use_blocks[i].id,
                        "content": str(result),
                    }
                )

        if tool_result_content:
            messages.append({"role": "user", "content": tool_result_content})


if __name__ == "__main__":
    # Check for required environment variables
    missing = []
    if not os.environ.get("COMPOSIO_API_KEY"):
        missing.append("COMPOSIO_API_KEY")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        missing.append("ANTHROPIC_API_KEY")

    if missing:
        print(f"Error: Missing required environment variables: {', '.join(missing)}")
        print("\nSet them before running:")
        for var in missing:
            print(f"  export {var}=your_key_here")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION
    result = run_philosophy_agent(question)
    print("\nResponse:")
    print("=" * 50)
    print(result)
