"""
Sophia - Philosophy Agent Web Chat
Tarayıcıda Sophia ile sohbet etmek için Flask uygulaması.

Kullanım:
  pip install -r requirements.txt
  export ANTHROPIC_API_KEY=sk-ant-...
  python app.py
  -> Tarayıcıda http://localhost:5000 aç
"""

import os
import sys
from flask import Flask, render_template, request, jsonify, session
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
inquiry — not a lecture, but a dialogue between two minds seeking understanding together.
You can speak both English and Turkish fluently."""

app = Flask(__name__)
app.secret_key = os.urandom(24)
client = anthropic.Anthropic()


@app.route("/")
def index():
    session["messages"] = []
    return render_template("chat.html")


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json
    user_message = data.get("message", "").strip()
    if not user_message:
        return jsonify({"error": "Boş mesaj"}), 400

    if "messages" not in session:
        session["messages"] = []

    messages = session["messages"]
    messages.append({"role": "user", "content": user_message})

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=SYSTEM_PROMPT,
        messages=messages,
    )

    reply = ""
    for block in response.content:
        if hasattr(block, "text"):
            reply = block.text
            break

    messages.append({"role": "assistant", "content": reply})
    session["messages"] = messages
    session.modified = True

    return jsonify({"reply": reply})


@app.route("/reset", methods=["POST"])
def reset():
    session["messages"] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("Hata: ANTHROPIC_API_KEY tanımlı değil.")
        print("  export ANTHROPIC_API_KEY=sk-ant-api03-...")
        sys.exit(1)
    print("Sophia başlatılıyor...")
    print("Tarayıcıda aç: http://localhost:5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
