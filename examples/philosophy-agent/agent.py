"""
Sophia - Philosophy Agent
An AI philosopher you can have ongoing conversations with.

Required environment variables:
  ANTHROPIC_API_KEY - Your Anthropic API key (https://console.anthropic.com)

Usage:
  python agent.py              # Start interactive chat with Sophia
  python agent.py "question"   # Ask a single question and exit
"""

import os
import sys

import anthropic

SYSTEM_PROMPT = """Your name is Sophia (from the Greek word for wisdom).
You are a philosopher with deep knowledge of philosophical traditions,
thinkers, and ideas from across history and cultures.

You explore philosophical questions with intellectual rigor, nuance, and curiosity.
You remember everything said in the conversation and build on previous exchanges.

When presented with a philosophical question or topic:
1. Draw on relevant philosophical traditions, schools of thought, and key thinkers
2. Present multiple perspectives and counterarguments
3. Examine underlying assumptions and logical implications
4. Synthesize insights into a coherent, thoughtful response
5. Ask thought-provoking follow-up questions to deepen the dialogue

You are warm, engaging, and genuinely curious. You treat conversation as collaborative
inquiry — not a lecture, but a dialogue between two minds seeking understanding together."""


def ask_sophia(client: anthropic.Anthropic, messages: list) -> str:
    """Send messages to Sophia and get her response."""
    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )
    for block in response.content:
        if hasattr(block, "text"):
            return block.text
    return ""


def chat_mode(client: anthropic.Anthropic) -> None:
    """Run an interactive conversation with Sophia."""
    print("\n" + "=" * 60)
    print("  Welcome. I am Sophia, your philosophical companion.")
    print("  Ask me anything. Type 'exit' or 'quit' to leave.")
    print("=" * 60 + "\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nSophia: Until we meet again. Keep questioning.")
            break

        if not user_input:
            continue

        if user_input.lower() in ("exit", "quit", "bye", "çıkış"):
            print("\nSophia: Until we meet again. Keep questioning.\n")
            break

        messages.append({"role": "user", "content": user_input})

        print("\nSophia: ", end="", flush=True)
        reply = ask_sophia(client, messages)
        print(reply)
        print()

        messages.append({"role": "assistant", "content": reply})


def single_question_mode(client: anthropic.Anthropic, question: str) -> None:
    """Ask Sophia a single question and print the answer."""
    print(f"\nYou: {question}\n")
    print("Sophia: ", end="", flush=True)
    reply = ask_sophia(client, [{"role": "user", "content": question}])
    print(reply)
    print()


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Error: ANTHROPIC_API_KEY is not set.")
        print("  export ANTHROPIC_API_KEY=your_key_here")
        sys.exit(1)

    client = anthropic.Anthropic()
    question = " ".join(sys.argv[1:])

    if question:
        single_question_mode(client, question)
    else:
        chat_mode(client)
