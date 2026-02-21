"""
Philosophy Agent - An AI agent that explores philosophical questions
using Anthropic Claude.

Required environment variables:
  ANTHROPIC_API_KEY - Your Anthropic API key (https://console.anthropic.com)

Optional environment variables:
  COMPOSIO_API_KEY  - Your Composio API key for tool integrations

Usage:
  python agent.py
  python agent.py "What is the meaning of life?"
"""

import os
import sys

import anthropic

SYSTEM_PROMPT = """You are a philosophy agent with deep knowledge of philosophical traditions,
thinkers, and ideas from across history and cultures. Your role is to explore philosophical
questions with intellectual rigor, nuance, and curiosity.

When presented with a philosophical question or topic:
1. Draw on relevant philosophical traditions, schools of thought, and key thinkers
2. Present multiple perspectives and counterarguments
3. Examine underlying assumptions and logical implications
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
    client = anthropic.Anthropic()

    print(f"Philosophy Agent\n{'=' * 50}")
    print(f"Question: {question}\n{'=' * 50}\n")

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": question}],
    )

    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: Missing required environment variable: ANTHROPIC_API_KEY")
        print("\n  export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    question = " ".join(sys.argv[1:]) if len(sys.argv) > 1 else DEFAULT_QUESTION
    result = run_philosophy_agent(question)
    print("\nResponse:")
    print("=" * 50)
    print(result)
