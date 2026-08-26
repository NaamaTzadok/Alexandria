# 🏛️ Alexandria - Autonomous Book Curation Agent

Alexandria is an autonomous, CLI-driven literary curator agent built with Python. Powered by Groq's high-speed inference engine, Alexandria implements a **ReAct (Reasoning + Acting) loop** that searches real-time bibliographic data via the **Open Library REST API**, filters titles semantically against user constraints, and tracks personal reading preferences across sessions in persistent local storage.

---

## 🏗️ Architecture & ReAct Loop

```text
               +-----------------------+
               |       CLI User        |
               +-----------------------+
                           |
                           v
              +-------------------------+
              |   Dynamic System Prompt | <--- (Injects user_memory.json)
              +-------------------------+
                           |
                           v
+-------------------> [Groq LLM] <--------------------+
|               (openai/gpt-oss-120b)                 |
|                          |                          |
|         +----------------+----------------+         |
|         |                                 |         |
|         v (Tool Call: search_books)       v (Tool Call: update_user_preference)
|  +--------------------+             +--------------------+
|  |  Open Library API  |             |  Local JSON Memory |
|  +--------------------+             +--------------------+
|         |                                 |         |
+---------+---------------------------------+---------+
                           |
               (Final Synthesis Output)
                           v
               +-----------------------+
               | Formatted Book Curation|
               +-----------------------+
```

## Key Features
* **Real-Time Data Grounding:** Eliminates hallucinations by validating titles, authors, page counts, and ratings using Open Library.
* **Autonomous Query Refinement:** Self-corrects search queries iteratively if initial calls return insufficient results.
* **Persistent Preference Memory:** Stores likes, dislikes, favorite genres, and custom notes in user_memory.json.
* **Resilience & Rate-Limiting:** Implements exponential backoff on HTTP/LLM timeouts and network connection pooling.
* **Token Usage Telemetry:** Tracks per-step and total turn token consumption via LLM usage metadata.

## Stack & Provider
* **LLM Provider:** [Groq](https://groq.com/) (OpenAI-compatible endpoint)
* **Model:** `openai/gpt-oss-120b`
* **Package & Dependency Manager:** `uv`
* **Data Sources:** Open Library Search API
* **Testing & Quality:** `pytest`, `unittest.mock`

## Installation & Setup
1. **Clone the repository:**
```Bash
git clone [https://github.com/your-username/alexandria.git](https://github.com/your-username/alexandria.git)
cd alexandria
```
2. **Install dependencies with `uv`:**
```Bash
uv sync
```
3. **Configure Environment Variables:**
Create a `.env` file in the root directory:
```.env
GROQ_API_KEY="gsk_your_groq_api_key_here"
```

## Usage
### Run the CLI Agent
```Bash
uv run agent.py
```
* Type your literary request directly in natural language.
* Type `/memory` to view your stored preferences.
* Type exit or quit to end the session.

### Run Unit Tests
```Bash
uv run pytest
```

### Run Behavior Evaluations
```Bash
uv run eval.py
```

## Agent Metrics & Performance
| Metric | Measured Value | Description |
| :--- | :---: | ---: |
| **Model** | `openai/gpt-oss-120b` | Base reasoning and function-calling LLM |
| **Average Turn Latency** | `1.4s - 2.8s` | End-to-end multi-step tool execution + synthesis |
| **Max Agent Iterations**	| `10 steps` | Safety boundary per conversational turn |
| **Context Window Consumption** | `~600 - 2,500 tokens` | Per turn including tool inputs and dynamic memory |
| **Evaluation Pass Rate** | `100% (5/5 tests)` | Verified across 5 synthetic behavioral evaluation turns |

## Documented Execution Runs
### Run 1: Conversational Turn (No Tool Required)
```Plaintext
You: Hello! What kind of books can you help me find today?

Alexandria:
Greetings! I am Alexandria, your personal literary curator. I can help you uncover 
compelling books tailored to your exact tastes—whether you're looking for atmospheric sci-fi, 
intricate psychological mysteries, historical epics, or concise standalone reads. 
Tell me what themes or moods you'd like to explore, and I will search our archives for you!

📊 [Token Usage] Total tokens consumed this turn: 412
```
### Run 2: Real-Time Book Search & Constraint Satisfaction (Tool Execution)
```Plaintext
You: I want a psychological sci-fi book about space exploration, under 400 pages, with no military action.

19:29:12 [INFO] alexandria: Step 0 Tokens -> Prompt: 461, Completion: 62, Total: 523
19:29:12 [INFO] alexandria: Step 0 | Tool: search_books | Args: {'query': 'psychological space exploration', 'max_results': 6}
19:29:13 [INFO] alexandria: Step 0 | Result: [{"title": "Sphere", "authors": ["Michael Crichton"], "page_count": 385, "average_rating": 4.19...}]
19:29:14 [INFO] alexandria: Step 1 Tokens -> Prompt: 742, Completion: 180, Total: 922

Alexandria:
Here are top recommendations that match your contemplative space exploration request:

| # | Title | Author | Pages | Avg Rating | Rationale |
|---|---|---|---|---|---|
| 1 | **Sphere** | Michael Crichton | 385 | 4.19 | Focuses on psychological tension and mental unraveling around an alien artifact rather than combat. |
| 2 | **Solaris** | Stanislaw Lem | 224 | 4.12 | A classic introspective study of human limits and isolation on an alien station. |

📊 [Token Usage] Total tokens consumed this turn: 1445
```

### Run 3: Memory Update & Dynamic Preference Storage
```Plaintext
You: I really dislike grimdark fantasy and heavy violence, please remember that.

19:35:01 [INFO] alexandria: Step 0 Tokens -> Prompt: 480, Completion: 44, Total: 524
19:35:01 [INFO] alexandria: Step 0 | Tool: update_user_preference | Args: {'category': 'disliked_books', 'item': 'grimdark fantasy and heavy violence'}
19:35:01 [INFO] alexandria: Step 0 | Result: Successfully added 'grimdark fantasy and heavy violence' to category 'disliked_books'.
19:35:02 [INFO] alexandria: Step 1 Tokens -> Prompt: 590, Completion: 58, Total: 648

Alexandria:
Understood. I have updated your reading profile to strictly exclude grimdark fantasy and books with heavy violence from all future recommendations.

You: /memory
{
  "liked_books": [],
  "disliked_books": [
    "grimdark fantasy and heavy violence"
  ],
  "favorite_genres": [],
  "notes": []
}
```