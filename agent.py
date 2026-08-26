"""
Alexandria - Autonomous Literary Curator Agent.

This module implements a CLI-based ReAct agent using OpenAI-compatible APIs (Groq)
and the Open Library API to search, verify, and curate personalized book recommendations.
State and user preferences are persisted locally in a JSON file.
"""

import json
import logging
import os
import time
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI, RateLimitError
import requests

# Configure application logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("alexandria")

# Silence noisy external libraries
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)

load_dotenv()

# Environment and model configuration
api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    log.error("GROQ_API_KEY environment variable is not set.")
    raise SystemExit("Missing GROQ_API_KEY. Please provide it in your .env file or environment.")

client = OpenAI(
    base_url="https://api.groq.com/openai/v1",
    api_key=api_key,
)
MODEL = "openai/gpt-oss-120b"
MEMORY_FILE = "user_memory.json"

DEFAULT_MEMORY: Dict[str, Any] = {
    "liked_books": [],
    "disliked_books": [],
    "favorite_genres": [],
    "notes": [],
}


# --- State & Memory Persistence ---

def load_memory() -> Dict[str, Any]:
    """
    Load user preferences and reading history from the local JSON storage.

    Returns:
        Dict[str, Any]: Dictionary containing stored user memory. Falls back
        to DEFAULT_MEMORY if the file does not exist or is corrupted.
    """
    if not os.path.exists(MEMORY_FILE):
        save_memory(DEFAULT_MEMORY)
        return DEFAULT_MEMORY.copy()

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                log.warning("Corrupted memory format detected. Resetting to defaults.")
                return DEFAULT_MEMORY.copy()
            return data
    except Exception as e:
        log.warning(f"Error loading memory file ({e}). Falling back to defaults.")
        return DEFAULT_MEMORY.copy()


def save_memory(memory_data: Dict[str, Any]) -> None:
    """
    Persist user preferences and memory state to the local JSON file.

    Args:
        memory_data (Dict[str, Any]): The memory structure to write to disk.
    """
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error(f"Failed to write memory to file: {e}")


# --- Agent Tools ---

def search_books(query: str, max_results: int = 6) -> str:
    """
    Query the Open Library REST API for book metadata matching given terms.

    Args:
        query (str): Keywords, title, author, or subject to search for.
        max_results (int, optional): Maximum number of search results. Defaults to 6.

    Returns:
        str: JSON-serialized list of matched book metadata, or an error/empty message.
    """
    url = "https://openlibrary.org/search.json"
    headers = {"User-Agent": "AlexandriaBookCurator/1.0 (educational-agent-project)"}
    params = {
        "q": query,
        "limit": max_results,
        "fields": "title,author_name,first_publish_year,number_of_pages_median,ratings_average,subject",
    }
    
    try:
        session = requests.Session()
        adapter = requests.adapters.HTTPAdapter(max_retries=2)
        session.mount("https://", adapter)

        response = session.get(url, params=params, headers=headers, timeout=8)
        if response.status_code != 200:
            log.warning(f"Open Library API returned status {response.status_code}")
            return f"Error: Open Library returned status {response.status_code}"

        data = response.json()
        docs = data.get("docs", [])

        if not docs:
            return f"No books found for query: '{query}'."

        book_results = []
        for doc in docs:
            subjects = doc.get("subject", [])
            subject_sample = ", ".join(subjects[:4]) if subjects else "N/A"
            book_results.append({
                "title": doc.get("title", "Unknown Title"),
                "authors": doc.get("author_name", ["Unknown Author"]),
                "first_publish_year": doc.get("first_publish_year", "N/A"),
                "page_count": doc.get("number_of_pages_median", "N/A"),
                "average_rating": round(doc.get("ratings_average", 0), 2) if doc.get("ratings_average") else "No rating",
                "topics": subject_sample,
            })
        return json.dumps(book_results, ensure_ascii=False)

    except requests.exceptions.Timeout:
        log.warning("Open Library request timed out.")
        return "Error: Request to Open Library timed out."
    except Exception as e:
        log.exception("Unexpected error occurred while querying Open Library API.")
        return f"Error querying Open Library API: {str(e)}"


def update_user_preference(category: str, item: str) -> str:
    """
    Store or update a specific preference item in the user's persistent profile.

    Args:
        category (str): Target category ('liked_books', 'disliked_books', 'favorite_genres', or 'notes').
        item (str): The preference or constraint to record.

    Returns:
        str: Status message describing the outcome of the update.
    """
    try:
        memory = load_memory()
        if category not in memory:
            memory[category] = []

        if isinstance(memory[category], list):
            if item not in memory[category]:
                memory[category].append(item)
                save_memory(memory)
                return f"Successfully added '{item}' to category '{category}'."
            return f"Item '{item}' already exists in category '{category}'."

        return f"Invalid category: '{category}'"
    except Exception as e:
        log.exception("Failed to update user preference.")
        return f"Error updating preference: {str(e)}"


# Tool mapping and schemas for function calling
AVAILABLE_TOOLS = {
    "search_books": search_books,
    "update_user_preference": update_user_preference,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "Searches Open Library by topic, genre, keywords, or authors, returning book metadata and subjects.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords (e.g. 'science fiction space', 'Asimov')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Maximum number of results to retrieve (default 6)",
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "update_user_preference",
            "description": "Saves a user preference into long-term memory.",
            "parameters": {
                "type": "object",
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": ["liked_books", "disliked_books", "favorite_genres", "notes"],
                        "description": "The category to update",
                    },
                    "item": {
                        "type": "string",
                        "description": "The specific preference or constraint to store",
                    },
                },
                "required": ["category", "item"],
            },
        },
    },
]


def build_system_prompt() -> str:
    """
    Construct the dynamic system prompt injecting the current persistent user memory.

    Returns:
        str: The formatted system prompt.
    """
    memory = load_memory()
    return f"""You are Alexandria, an enlightened literary curator.
Your mission is to recommend verified books by retrieving real-time data from Open Library.

Current User Memory:
{json.dumps(memory, indent=2, ensure_ascii=False)}

CRITICAL RULES:
1. NEVER recommend a book from your internal training memory alone.
2. When the user asks for book recommendations, topics, or authors, you MUST FIRST call `search_books`. For simple greetings, questions about your capabilities, or casual conversation, respond conversationally without calling search tools.
3. Perform at most 2-3 search queries to find suitable candidates before synthesizing your response.
4. Only recommend books that were returned in the `search_books` tool output. Use the exact titles, page counts, and Open Library average ratings provided by the tool.
5. If the user expresses preferences, dislikes, or constraints, call `update_user_preference` to store them.
6. Provide 2-3 concise recommendations based strictly on the retrieved results with clear rationales.
"""

def call_with_retry(messages, tools, max_retries=4):
    """Call the OpenAI API with exponential backoff on rate limit errors."""
    for attempt in range(max_retries):
        try:
            return client.chat.completions.create(model=MODEL, messages=messages, tools=tools)
        except RateLimitError:
            wait = 2 ** attempt          # 1, 2, 4, 8 seconds
            print(f"rate limit, waiting {wait} seconds")
            time.sleep(wait)
    raise RuntimeError("failed after all attempts")

def run_alexandria_cli(max_steps: int = 10) -> None:
    """
    Execute the interactive CLI loop for Alexandria with token usage tracking.

    Args:
        max_steps (int, optional): Maximum tool execution iterations per user prompt. Defaults to 10.
    """
    print("=" * 60)
    print("🏛️  Welcome to Alexandria! (Type 'exit' or 'quit' to stop)")
    print("=" * 60)

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": build_system_prompt()}
    ]

    while True:
        try:
            user_input = input("\nYou: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ["exit", "quit"]:
                print("\nGoodbye, happy reading! 📖")
                break
            if user_input.lower() == "/memory":
                print(json.dumps(load_memory(), indent=2, ensure_ascii=False))
                continue

            messages.append({"role": "user", "content": user_input})
            turn_total_tokens = 0

            # ReAct Execution Loop
            for step in range(max_steps):
                try:
                    response = call_with_retry(messages, TOOLS_SCHEMA)
                except Exception as e:
                    print(f"\n[Connection Error] Failed to communicate with LLM provider: {e}")
                    log.error(f"Provider API call failed: {e}")
                    break

                # Extract and log token usage for this step
                usage = getattr(response, "usage", None)
                if usage:
                    step_tokens = usage.total_tokens
                    turn_total_tokens += step_tokens
                    log.info(
                        f"Step {step} Tokens -> Prompt: {usage.prompt_tokens}, "
                        f"Completion: {usage.completion_tokens}, Total: {step_tokens}"
                    )

                msg = response.choices[0].message
                messages.append(msg)

                # Termination condition: model provided final textual response
                if not msg.tool_calls:
                    print(f"\nAlexandria:\n{msg.content}")
                    print(f"\n📊 [Token Usage] Total tokens consumed this turn: {turn_total_tokens}")
                    break

                for call in msg.tool_calls:
                    fn_name = call.function.name
                    raw_args = call.function.arguments

                    fn = AVAILABLE_TOOLS.get(fn_name)
                    if not fn:
                        result = f"Error: Tool '{fn_name}' is not recognized."
                        log.warning(result)
                    else:
                        try:
                            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                            log.info(f"Step {step} | Tool: {fn_name} | Args: {args}")
                            result = fn(**args)
                        except json.JSONDecodeError:
                            result = f"Error: Failed to parse JSON arguments for tool '{fn_name}'."
                            log.error(f"Invalid JSON in tool call arguments: {raw_args}")
                        except Exception as e:
                            result = f"Error while running '{fn_name}': {e}"
                            log.exception(f"Execution error in tool {fn_name}")

                    log.info(f"Step {step} | Result: {str(result)[:100]}...")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(result),
                    })
            else:
                print("\nAlexandria: Reached the maximum execution steps without producing a final answer.")
                print(f"📊 [Token Usage] Total tokens consumed across {max_steps} steps: {turn_total_tokens}")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as err:
            log.exception(f"Unhandled CLI exception: {err}")
            print(f"\n[Error] Unexpected error: {err}")

if __name__ == "__main__":
    run_alexandria_cli()