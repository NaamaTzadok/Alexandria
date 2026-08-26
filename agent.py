import json
import logging
import os
from typing import Any, Dict, List
from dotenv import load_dotenv
from openai import OpenAI
import requests

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("agent")

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")
if not api_key:
    log.error("GROQ_API_KEY environment variable is missing.")
    raise SystemExit("Please set GROQ_API_KEY in your .env file or environment.")

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


# --- Memory Management Functions ---

def load_memory() -> Dict[str, Any]:
    """Load user preferences and history from a local JSON file with fallback handling."""
    if not os.path.exists(MEMORY_FILE):
        save_memory(DEFAULT_MEMORY)
        return DEFAULT_MEMORY.copy()

    try:
        with open(MEMORY_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if not isinstance(data, dict):
                return DEFAULT_MEMORY.copy()
            return data
    except Exception as e:
        log.warning(f"Failed to load memory file ({e}), resetting to default structure.")
        return DEFAULT_MEMORY.copy()


def save_memory(memory_data: Dict[str, Any]) -> None:
    """Save updated memory to the local JSON file safely."""
    try:
        with open(MEMORY_FILE, "w", encoding="utf-8") as f:
            json.dump(memory_data, f, indent=2, ensure_ascii=False)
    except Exception as e:
        log.error(f"Failed to save memory file: {e}")


# --- Tool Implementations ---

def search_books(query: str, max_results: int = 6) -> str:
    """Search Open Library API for books matching a title, author, or subject keyword."""
    url = "https://openlibrary.org/search.json"
    headers = {"User-Agent": "AlexandriaBookCurator/1.0 (educational-project)"}
    params = {
        "q": query,
        "limit": max_results,
        "fields": "title,author_name,first_publish_year,number_of_pages_median,ratings_average,subject",
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=10)
        if response.status_code != 200:
            log.error(f"Open Library API returned status {response.status_code}: {response.text[:200]}")
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
        return "Error: Request to Open Library timed out."
    except Exception as e:
        log.exception("Unexpected error in search_books")
        return f"Error querying Open Library API: {str(e)}"


def update_user_preference(category: str, item: str) -> str:
    """Update user preference profile safely."""
    try:
        memory = load_memory()
        if category not in memory:
            memory[category] = []

        if isinstance(memory[category], list):
            if item not in memory[category]:
                memory[category].append(item)
                save_memory(memory)
                return f"Successfully added '{item}' to {category}."
            return f"'{item}' is already in {category}."

        return f"Invalid category: {category}"
    except Exception as e:
        return f"Error updating preference: {str(e)}"


AVAILABLE_TOOLS = {
    "search_books": search_books,
    "update_user_preference": update_user_preference,
}

TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "search_books",
            "description": "Searches Open Library by topic, genre, keywords, or authors, returning volume details and topics.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "Search query keywords (e.g. 'science fiction space', 'Asimov')",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to retrieve (default 6)",
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
                        "description": "The value to store",
                    },
                },
                "required": ["category", "item"],
            },
        },
    },
]


def build_system_prompt() -> str:
    """Construct dynamic system prompt with current user memory injected."""
    memory = load_memory()
    return f"""You are Alexandria, an enlightened literary curator.
Your mission is to recommend verified books by retrieving real-time data from Open Library.

Current User Memory:
{json.dumps(memory, indent=2, ensure_ascii=False)}

CRITICAL RULES:
1. NEVER recommend a book from your internal training memory alone.
2. For EVERY user request, you MUST FIRST call `search_books` with a tailored query.
3. Perform at most 2-3 search queries to find suitable candidates before synthesizing your response.
4. Only recommend books that were returned in the `search_books` tool output. Use the exact titles, page counts, and Open Library average ratings provided by the tool.
5. If the user expresses preferences, dislikes, or constraints, call `update_user_preference` to store them.
6. Provide 2-3 concise recommendations based strictly on the retrieved results with clear rationales.
"""


def run_alexandria_cli(max_steps: int = 10) -> None:
    """Interactive CLI loop for Alexandria with complete error containment."""
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

            for step in range(max_steps):
                try:
                    response = client.chat.completions.create(
                        model=MODEL,
                        messages=messages,
                        tools=TOOLS_SCHEMA,
                    )
                except Exception as e:
                    print(f"\n[Connection Error] Failed to communicate with LLM provider: {e}")
                    break

                msg = response.choices[0].message
                messages.append(msg)

                if not msg.tool_calls:
                    print(f"\nAlexandria:\n{msg.content}")
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
                            log.info(f"Step {step} | Tool: {fn_name} | Arguments: {args}")
                            result = fn(**args)
                        except json.JSONDecodeError:
                            result = f"Error: Failed to parse JSON arguments for tool '{fn_name}'."
                        except Exception as e:
                            result = f"Error while running '{fn_name}': {e}"

                    log.info(f"Step {step} | Result: {str(result)[:100]}")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": call.id,
                        "content": str(result),
                    })
            else:
                print("\nAlexandria: I have reached the maximum number of steps without a final answer.")

        except KeyboardInterrupt:
            print("\nSession ended.")
            break
        except Exception as err:
            log.exception(f"Unexpected CLI error: {err}")
            print(f"\n[Error] Something unexpected happened: {err}")


if __name__ == "__main__":
    run_alexandria_cli()