"""
Philosophy Agent — powered by Composio + Anthropic Claude

An agentic philosophy assistant that engages in deep philosophical discourse.
It can look up philosophical references and run an interactive conversation loop.

Usage:
    export ANTHROPIC_API_KEY=<your-key>
    export COMPOSIO_API_KEY=<your-key>   # optional, enables web-search tools
    python agent.py
"""

import os
import sys

import anthropic
from anthropic.types import Message, TextBlock, ToolUseBlock

SYSTEM_PROMPT = """You are Philosopher, a rigorous and engaging AI philosopher. You:

- Draw on the full breadth of Western and Eastern philosophical traditions
  (Socrates, Plato, Aristotle, Kant, Hegel, Nietzsche, Wittgenstein, Heidegger,
   Confucius, Nagarjuna, Zhuangzi, and many more)
- Engage Socratically — you ask clarifying questions to sharpen arguments
- Distinguish carefully between metaphysics, epistemology, ethics, logic, and aesthetics
- Present multiple philosophical perspectives on any given question, including
  ones that challenge the interlocutor's assumptions
- Cite philosophers and primary texts when relevant (e.g., "As Kant argues in the
  Critique of Pure Reason…")
- Acknowledge genuine uncertainty and the limits of philosophical knowledge
- Translate abstract concepts into concrete, relatable examples

When the user raises a topic, explore it thoroughly: identify the core question,
survey major positions, raise objections, and invite the user to defend or refine
their view. Philosophy is a dialogue, not a lecture.
"""

WELCOME_BANNER = """
╔══════════════════════════════════════════════════════════╗
║              Philosophy Agent  (Composio + Claude)       ║
║                                                          ║
║  "The unexamined life is not worth living." — Socrates   ║
╚══════════════════════════════════════════════════════════╝

Type a philosophical question or topic to begin.
Type 'quit' or 'exit' to end the session.
"""


def build_composio_tools() -> list:
    """
    Attempt to load Composio tools (web-search) for the agent.
    Falls back gracefully if Composio is not installed or no API key is set.
    """
    try:
        from composio import Composio  # type: ignore
        from composio_anthropic import AnthropicProvider  # type: ignore

        api_key = os.environ.get("COMPOSIO_API_KEY")
        if not api_key:
            return []

        composio = Composio(provider=AnthropicProvider(), api_key=api_key)
        # Prefer TAVILY search; fall back to an empty list if unavailable
        try:
            tools = composio.tools.get(
                user_id="default",
                toolkits=["TAVILY"],
            )
            print(f"[composio] Loaded {len(tools)} tool(s) from Composio.\n")
            return tools
        except Exception:
            return []
    except ImportError:
        return []


def handle_tool_calls(
    composio_provider,
    response: Message,
    user_id: str = "default",
) -> list:
    """
    Execute any tool calls present in `response` via the Composio provider
    and return the list of tool-result message dicts for the next API call.
    """
    if composio_provider is None:
        return []

    tool_results = []
    for block in response.content:
        if isinstance(block, ToolUseBlock):
            result = composio_provider.execute_tool_call(
                user_id=user_id,
                tool_call=block,
            )
            tool_results.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": str(result),
                        }
                    ],
                }
            )
    return tool_results


def run_agent_turn(
    client: anthropic.Anthropic,
    messages: list,
    tools: list,
    composio_provider=None,
) -> str:
    """
    Run a single agent turn (with optional tool-use loop) and return the
    final text response.
    """
    model = "claude-opus-4-5" if not os.environ.get("ANTHROPIC_MODEL") else os.environ["ANTHROPIC_MODEL"]

    kwargs: dict = {
        "model": model,
        "max_tokens": 2048,
        "system": SYSTEM_PROMPT,
        "messages": messages,
    }
    if tools:
        kwargs["tools"] = tools

    # Agentic tool-use loop
    while True:
        response: Message = client.messages.create(**kwargs)

        # Append assistant turn to history
        messages.append({"role": "assistant", "content": response.content})

        # If no tool calls, we are done
        if response.stop_reason != "tool_use":
            break

        # Execute tool calls and add results
        tool_result_messages = handle_tool_calls(composio_provider, response)
        if not tool_result_messages:
            # Safety: if we can't execute tools, break to avoid infinite loop
            break
        messages.extend(tool_result_messages)

    # Extract final text
    text_parts = [block.text for block in response.content if isinstance(block, TextBlock)]
    return "\n".join(text_parts).strip()


def main() -> None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        print("Error: ANTHROPIC_API_KEY environment variable is not set.")
        sys.exit(1)

    client = anthropic.Anthropic(api_key=api_key)

    # Load Composio tools (optional)
    tools = build_composio_tools()

    # Retrieve provider object for tool execution (if composio loaded)
    composio_provider = None
    if tools:
        try:
            from composio import Composio  # type: ignore
            from composio_anthropic import AnthropicProvider  # type: ignore

            composio_provider = Composio(
                provider=AnthropicProvider(),
                api_key=os.environ.get("COMPOSIO_API_KEY"),
            ).provider
        except ImportError:
            tools = []

    print(WELCOME_BANNER)

    conversation: list = []

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\nFarewell. Keep questioning.")
            break

        if not user_input:
            continue

        if user_input.lower() in {"quit", "exit", "q"}:
            print("\nFarewell. Keep questioning.")
            break

        conversation.append({"role": "user", "content": user_input})

        try:
            reply = run_agent_turn(
                client=client,
                messages=conversation,
                tools=tools,
                composio_provider=composio_provider,
            )
        except anthropic.APIError as exc:
            print(f"\n[API error] {exc}\n")
            # Remove the failed user turn so the conversation stays consistent
            conversation.pop()
            continue

        print(f"\nPhilosopher: {reply}\n")


if __name__ == "__main__":
    main()
