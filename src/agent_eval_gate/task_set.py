"""Task set definitions (≥16 tasks across 4 types)."""

from __future__ import annotations

from agent_eval_gate.protocols import Task

# In the style of GAIA (multi-step), Tau-Bench (tool-use), SWE-bench (coding),
# and general faithfulness/RAG tasks.


def load_task_set() -> list[Task]:
    return [
        # ── QA (4 tasks) ──────────────────────────────────────────────────────
        Task(
            id="qa-01",
            type="qa",
            prompt="What is the capital of France? Provide only the city name.",
            expected="Paris",
            judge_criteria="The answer must be the exact city name 'Paris'.",
        ),
        Task(
            id="qa-02",
            type="qa",
            prompt="Who wrote 'Pride and Prejudice'? Provide only the author's full name.",
            expected="Jane Austen",
            judge_criteria="The answer must be the exact author name 'Jane Austen'.",
        ),
        Task(
            id="qa-03",
            type="qa",
            prompt="What is 15 * 27? Provide only the integer result.",
            expected="405",
            judge_criteria="The answer must be the exact integer '405'.",
        ),
        Task(
            id="qa-04",
            type="qa",
            prompt="In what year did the Titanic sink? Provide only the year.",
            expected="1912",
            judge_criteria="The answer must be the exact year '1912'.",
        ),
        # ── Structured / typed output (4 tasks) ───────────────────────────────
        Task(
            id="structured-01",
            type="structured",
            prompt="Return a JSON object with keys: product (string), price (number), in_stock (boolean). Product is 'Widget', price is 19.99, in_stock is true.",
            expected={"product": "Widget", "price": 19.99, "in_stock": True},
            judge_criteria="The output must be valid JSON with the exact keys and values specified.",
        ),
        Task(
            id="structured-02",
            type="structured",
            prompt="Return a JSON object with keys: title (string), author (string), year (integer). Title is 'Dune', author is 'Frank Herbert', year is 1965.",
            expected={"title": "Dune", "author": "Frank Herbert", "year": 1965},
            judge_criteria="The output must be valid JSON with the exact keys and values specified.",
        ),
        Task(
            id="structured-03",
            type="structured",
            prompt="Return a JSON array of 3 strings: red, green, blue.",
            expected=["red", "green", "blue"],
            judge_criteria="The output must be a JSON array of exactly three strings in the specified order.",
        ),
        Task(
            id="structured-04",
            type="structured",
            prompt="Return a JSON object with keys: event (string), date (YYYY-MM-DD), attendees (integer). Event is 'Launch', date is '2026-01-15', attendees is 120.",
            expected={"event": "Launch", "date": "2026-01-15", "attendees": 120},
            judge_criteria="The output must be valid JSON with the exact keys and values specified.",
        ),
        # ── Tool use / function calling (4 tasks) ─────────────────────────────
        Task(
            id="tool_use-01",
            type="tool_use",
            prompt="Use the available tool 'get_weather' with city='Tokyo' and return the temperature.",
            expected=None,  # judge checks tool call correctness + plausible temp
            context="The get_weather tool returns a JSON with city and temperature_celsius. Tokyo is typically 10-30C.",
            tools=[
                {
                    "name": "get_weather",
                    "description": "Get current temperature for a city.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {"type": "string", "description": "City name"}
                        },
                        "required": ["city"],
                    },
                }
            ],
            judge_criteria="The agent must call the get_weather tool with city='Tokyo' and return a plausible temperature value.",
        ),
        Task(
            id="tool_use-02",
            type="tool_use",
            prompt="Use the available tool 'calculator' to compute 144 / 12. Return only the numeric result.",
            expected="12",
            tools=[
                {
                    "name": "calculator",
                    "description": "Evaluate a math expression.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {"type": "string", "description": "Math expression"}
                        },
                        "required": ["expression"],
                    },
                }
            ],
            judge_criteria="The agent must call the calculator tool with expression='144 / 12' and return 12.",
        ),
        Task(
            id="tool_use-03",
            type="tool_use",
            prompt="Use the available tool 'search' to find the tallest mountain and return its name.",
            expected="Mount Everest",
            tools=[
                {
                    "name": "search",
                    "description": "Search the knowledge base.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "query": {"type": "string", "description": "Search query"}
                        },
                        "required": ["query"],
                    },
                }
            ],
            context="The search tool returns the best matching fact. Mount Everest is the tallest mountain above sea level.",
            judge_criteria="The agent must call the search tool and return 'Mount Everest' as the tallest mountain.",
        ),
        Task(
            id="tool_use-04",
            type="tool_use",
            prompt="Use the available tool 'convert_currency' to convert 100 USD to EUR. Return the amount and currency.",
            expected="EUR",
            tools=[
                {
                    "name": "convert_currency",
                    "description": "Convert an amount between currencies.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "amount": {"type": "number"},
                            "from_currency": {"type": "string"},
                            "to_currency": {"type": "string"},
                        },
                        "required": ["amount", "from_currency", "to_currency"],
                    },
                }
            ],
            judge_criteria="The agent must call convert_currency with amount=100, from_currency='USD', to_currency='EUR' and return the EUR amount.",
        ),
        # ── Faithfulness (4 tasks) ────────────────────────────────────────────
        Task(
            id="faithfulness-01",
            type="faithfulness",
            prompt="Using only the context below, answer: What causes seasons on Earth?\nContext: Seasons are caused by the tilt of Earth's axis as it orbits the Sun. This tilt means different hemispheres receive varying amounts of sunlight throughout the year.",
            expected="The tilt of Earth's axis as it orbits the Sun",
            context="Seasons are caused by the tilt of Earth's axis as it orbits the Sun. This tilt means different hemispheres receive varying amounts of sunlight throughout the year.",
            judge_criteria="The answer must be supported by the provided context and must not introduce unsupported claims.",
        ),
        Task(
            id="faithfulness-02",
            type="faithfulness",
            prompt="Using only the context below, answer: What is the boiling point of water at sea level?\nContext: At standard atmospheric pressure (sea level), pure water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
            expected="100 degrees Celsius",
            context="At standard atmospheric pressure (sea level), pure water boils at 100 degrees Celsius (212 degrees Fahrenheit).",
            judge_criteria="The answer must be supported by the provided context and must not introduce unsupported claims.",
        ),
        Task(
            id="faithfulness-03",
            type="faithfulness",
            prompt="Using only the context below, answer: Who invented the World Wide Web?\nContext: Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN.",
            expected="Tim Berners-Lee",
            context="Tim Berners-Lee invented the World Wide Web in 1989 while working at CERN.",
            judge_criteria="The answer must be supported by the provided context and must not introduce unsupported claims.",
        ),
        Task(
            id="faithfulness-04",
            type="faithfulness",
            prompt="Using only the context below, answer: What planet is known as the Red Planet?\nContext: Mars is often called the Red Planet because iron oxide on its surface gives it a reddish appearance.",
            expected="Mars",
            context="Mars is often called the Red Planet because iron oxide on its surface gives it a reddish appearance.",
            judge_criteria="The answer must be supported by the provided context and must not introduce unsupported claims.",
        ),
    ]
