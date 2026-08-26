"""
Evaluation script for Alexandria Agent.
Runs 5 distinct behavioral evaluations and prints pass/fail summary.
"""

import json
import os
from typing import Any, Callable, Dict, List
from agent import (
    AVAILABLE_TOOLS,
    MODEL,
    TOOLS_SCHEMA,
    build_system_prompt,
    client,
    load_memory,
    save_memory,
)

EVAL_MEMORY_FILE = "eval_memory.json"


def setup_eval_env() -> None:
    """Prepare a isolated memory state for evaluations."""
    initial_memory = {
        "liked_books": ["Foundation"],
        "disliked_books": ["Twilight"],
        "favorite_genres": ["science fiction"],
        "notes": ["dislikes intense gore and grimdark"]
    }
    with open(EVAL_MEMORY_FILE, "w", encoding="utf-8") as f:
        json.dump(initial_memory, f, indent=2)


def teardown_eval_env() -> None:
    """Cleanup temporary evaluation files."""
    if os.path.exists(EVAL_MEMORY_FILE):
        os.remove(EVAL_MEMORY_FILE)


def run_agent_turn(prompt: str) -> Dict[str, Any]:
    """Execute a single agent turn and collect executed tools and response."""
    messages = [
        {"role": "system", "content": build_system_prompt()},
        {"role": "user", "content": prompt}
    ]
    tools_called = []
    final_content = ""

    for _ in range(6):
        response = client.chat.completions.create(
            model=MODEL,
            messages=messages,
            tools=TOOLS_SCHEMA
        )
        msg = response.choices[0].message
        messages.append(msg)

        if not msg.tool_calls:
            final_content = msg.content or ""
            break

        for call in msg.tool_calls:
            tools_called.append({
                "name": call.function.name,
                "args": json.loads(call.function.arguments)
            })
            fn = AVAILABLE_TOOLS.get(call.function.name)
            result = fn(**json.loads(call.function.arguments)) if fn else "Tool not found"
            messages.append({"role": "tool", "tool_call_id": call.id, "content": str(result)})

    return {
        "final_content": final_content,
        "tools_called": tools_called,
        "memory": load_memory()
    }


# --- Evaluation Test Cases ---

def eval_tool_usage_on_book_query() -> bool:
    """Eval 1: Book search must trigger search_books tool."""
    res = run_agent_turn("Can you recommend 2 philosophical sci-fi books?")
    return any(t["name"] == "search_books" for t in res["tools_called"])


def eval_memory_update_on_preference() -> bool:
    """Eval 2: Stating explicit dislike must trigger memory update tool."""
    res = run_agent_turn("I really hate zombie apocalypse survival tropes, never recommend them.")
    memory_updated = any(t["name"] == "update_user_preference" for t in res["tools_called"])
    return memory_updated


def eval_conversational_response() -> bool:
    """Eval 3: Casual greeting should not trigger search tools."""
    res = run_agent_turn("Hello! Who are you and how can you help me?")
    no_search_called = not any(t["name"] == "search_books" for t in res["tools_called"])
    return no_search_called and len(res["final_content"]) > 10


def eval_search_query_relevance() -> bool:
    """Eval 4: Query formulation contains requested domain concepts."""
    res = run_agent_turn("Find speculative fiction books regarding artificial intelligence ethics.")
    for tool in res["tools_called"]:
        if tool["name"] == "search_books":
            query = tool["args"].get("query", "").lower()
            if "ai" in query or "intelligence" in query or "speculative" in query or "ethics" in query:
                return True
    return False


def eval_output_format_and_rationales() -> bool:
    """Eval 5: Recommendations must provide book titles and clear rationales."""
    res = run_agent_turn("Suggest a cyberpunk novel under 300 pages.")
    text = res["final_content"].lower()
    return ("pages" in text or "rating" in text or "why" in text) and len(text) > 40


def run_all_evals() -> None:
    evals: List[tuple[str, Callable[[], bool]]] = [
        ("Tool Trigger on Search Request", eval_tool_usage_on_book_query),
        ("Persistent Memory Update Trigger", eval_memory_update_on_preference),
        ("Zero-Tool Handling on Greeting", eval_conversational_response),
        ("Contextual Query Formulation", eval_search_query_relevance),
        ("Structured Rationale in Recommendation", eval_output_format_and_rationales),
    ]

    print("=" * 60)
    print("🧪 Running Alexandria Agent Behavior Evaluations...")
    print("=" * 60)

    passed_count = 0
    for idx, (name, eval_fn) in enumerate(evals, start=1):
        try:
            passed = eval_fn()
        except Exception as e:
            print(f"[{idx}/5] ❌ {name} -> Error: {e}")
            continue

        if passed:
            passed_count += 1
            print(f"[{idx}/5] ✅ PASS: {name}")
        else:
            print(f"[{idx}/5] ❌ FAIL: {name}")

    total = len(evals)
    success_rate = (passed_count / total) * 100
    print("-" * 60)
    print(f"📊 Summary: {passed_count}/{total} Passed ({success_rate:.1f}%)")
    print("=" * 60)


if __name__ == "__main__":
    run_all_evals()