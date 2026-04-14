"""Switch node with multiple branches.

Demonstrates a multi-way switch (more than two branches) for routing
based on detected values. Uses a decision node with many options to
model a switch/case pattern. Executed with a real LLM via the runner.

Usage:
    export ANTHROPIC_API_KEY="sk-ant-..."   # or OPENAI_API_KEY
    uv run examples/07_switch_router.py
"""

from __future__ import annotations

from quartermaster_graph import Graph
from _runner import run_graph

# Multi-language support agent with switch-style routing
agent = (
    Graph("Multi-Language Agent")
    .start()
    .user("Enter your message")
    .instruction("Detect language", model="claude-haiku-4-5-20251001", system_instruction="Detect the language. Output: en/es/fr/de/other")
    .decision("Language?", options=["en", "es", "fr", "de", "other"])
    .on("en").instruction("English handler", model="claude-haiku-4-5-20251001", system_instruction="Respond in English").end()
    .on("es").instruction("Spanish handler", model="claude-haiku-4-5-20251001", system_instruction="Responde en espanol").end()
    .on("fr").instruction("French handler", model="claude-haiku-4-5-20251001", system_instruction="Repondez en francais").end()
    .on("de").instruction("German handler", model="claude-haiku-4-5-20251001", system_instruction="Antworten Sie auf Deutsch").end()
    .on("other").instruction("Fallback", model="claude-haiku-4-5-20251001", system_instruction="Respond in English, note language").end()
    # No merge -- decision picks one language branch, they converge on End.
    .end()
)

run_graph(agent, user_input="Bonjour, comment allez-vous aujourd'hui?")
