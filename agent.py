import json
from groq import Groq
from config import GROQ_API_KEY, LLM_MODEL, MAX_TOOL_ROUNDS
from tools import lookup_plant, get_seasonal_conditions

_client = Groq(api_key=GROQ_API_KEY)

# ──────────────────────────────────────────────
# Tool definitions
#
# These are the schemas that tell the LLM what tools are available and how to
# call them. The LLM reads these descriptions and decides when (and how) to use
# each tool. They're already complete — your job is to implement the tool
# functions in tools.py and the agent loop below.
# ──────────────────────────────────────────────

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "lookup_plant",
            "description": (
                "Look up care information for a specific houseplant by name. "
                "Returns detailed watering, light, humidity, and temperature requirements. "
                "Use this whenever the user asks about a specific plant."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "plant_name": {
                        "type": "string",
                        "description": "The plant name to look up. Can be a common name, scientific name, or nickname (e.g., 'pothos', 'devil's ivy', 'Monstera deliciosa').",
                    }
                },
                "required": ["plant_name"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_seasonal_conditions",
            "description": (
                "Get seasonal care adjustments for houseplants. "
                "Returns guidance on watering, fertilizing, light, and pests for the current or specified season. "
                "Use this when a user asks a season-specific question, or to complement plant care advice with seasonal context."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "season": {
                        "type": "string",
                        "description": "The season to get care conditions for. If omitted, the current season is detected automatically.",
                        "enum": ["spring", "summer", "fall", "winter"],
                    }
                },
                "required": [],
            },
        },
    },
]

# ──────────────────────────────────────────────
# System prompt
# ──────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are a knowledgeable and friendly plant care advisor. "
    "Help users care for their houseplants by looking up specific plant information "
    "and current seasonal conditions using your available tools.\n\n"
    "Always use your tools to look up plant-specific information before answering — "
    "don't rely on your general knowledge alone. If a plant isn't in your database, "
    "say so clearly and offer general guidance based on what the user describes. "
    "The general guidance should be useful to the user, but more importantly, accurate.\n\n"
    "Keep your advice practical and specific. Cite the source of your information "
    "when you have it (e.g., 'According to the care data for your monstera...')."
)

# ──────────────────────────────────────────────
# Tool dispatch
#
# This is already complete. It routes tool calls from the LLM to the actual
# Python functions in tools.py, and returns results as JSON strings (which is
# what the Groq API expects for tool results).
# ──────────────────────────────────────────────


def dispatch_tool(tool_name: str, tool_args: dict) -> str:
    """Route a tool call to the correct function and return the result as a JSON string."""
    print(f"  → Tool call: {tool_name}({tool_args})")
    if tool_name == "lookup_plant":
        result = lookup_plant(tool_args["plant_name"])
    elif tool_name == "get_seasonal_conditions":
        result = get_seasonal_conditions(tool_args.get("season"))
    else:
        result = {"error": f"Unknown tool: {tool_name}"}
    print(
        f"  ← Result: {json.dumps(result)[:120]}{'...' if len(json.dumps(result)) > 120 else ''}"
    )
    return json.dumps(result)


# ──────────────────────────────────────────────
# Agent loop
# ──────────────────────────────────────────────


def run_agent(user_message: str, history: list) -> str:
    """
    Run the plant care agent for one user turn and return its response.

    ☑️ — Milestone 2:

    The agent loop follows a specific pattern that you'll implement here. Read
    specs/agent-loop-spec.md carefully before writing any code — understand the
    full loop before implementing any part of it.

    The loop works like this:
      1. Build a messages list: system prompt + conversation history + new user message
      2. Call the LLM with messages and TOOL_DEFINITIONS
      3. If the response contains tool_calls:
           a. Append the assistant message (with tool_calls) to messages
           b. For each tool call: execute via dispatch_tool(), append the result
           c. Call the LLM again with the updated messages
           d. Repeat until no more tool_calls (or MAX_TOOL_ROUNDS is reached)
      4. Return the final text response

    Key details to get right:
      - The assistant message must be appended BEFORE tool results
      - Tool result messages use role="tool" with a tool_call_id field
      - Append the assistant's message object directly (not just its content)
      - The history format from Gradio: list of [user_message, assistant_message] pairs

    Before writing code, complete specs/agent-loop-spec.md.
    """
    # 1. Create messages list
    # msgs has system prompt, history (usr, assistant, so on), new usr msg
    msgs = [{"role": "system", "content": SYSTEM_PROMPT}]

    def _extract_text(raw_content):
        if raw_content is None:
            return ""
        if isinstance(raw_content, str):
            return raw_content
        if isinstance(raw_content, list) and raw_content:
            first = raw_content[0]
            if isinstance(first, dict):
                return first.get("text", "")
            if isinstance(first, str):
                return first
        return ""

    for hist_item in history:
        if isinstance(hist_item, (list, tuple)) and len(hist_item) >= 2:
            msgs.append({"role": "user", "content": _extract_text(hist_item[0])})
            if hist_item[1] is not None:
                msgs.append(
                    {"role": "assistant", "content": _extract_text(hist_item[1])}
                )
        elif isinstance(hist_item, dict):
            role = hist_item.get("role")
            if role in {"user", "assistant"}:
                msgs.append(
                    {"role": role, "content": _extract_text(hist_item.get("content"))}
                )

    msgs.append({"role": "user", "content": user_message})

    # 2. Call the LLM, retrying once without tool use on errors,
    # then handling gracefully
    def _create_completion(messages, use_tools=True):
        kwargs = {"model": LLM_MODEL, "messages": messages}
        if use_tools:
            kwargs["tools"] = TOOL_DEFINITIONS
            kwargs["tool_choice"] = "auto"
        else:
            kwargs["tool_choice"] = "none"
        return _client.chat.completions.create(**kwargs)

    def _is_400_error(exc):
        status = (
            getattr(exc, "status_code", None)
            or getattr(exc, "status", None)
            or getattr(exc, "code", None)
        )
        if status == 400 or status == "400":
            return True
        message = str(exc).lower()
        return "400" in message or "bad request" in message

    try:
        res = _create_completion(msgs, use_tools=True)
    except Exception as exc:
        if _is_400_error(exc):
            try:
                res = _create_completion(msgs, use_tools=False)
                return (
                    "I couldn't use the plant lookup tools right now, "
                    "but I can still answer from general plant care knowledge.\n\n"
                    + res.choices[0].message.content
                )
            except Exception:
                return (
                    "Sorry, the plant advisor is temporarily unavailable. "
                    "Please try again in a moment."
                )
        return (
            "Sorry, the plant advisor is temporarily unavailable. "
            "Please try again in a moment."
        )

    i = 0

    # 3. Run the agent loop
    while (res_msg := res.choices[0].message).tool_calls and (
        i := i + 1
    ) <= MAX_TOOL_ROUNDS:
        print(f"{i=} out of max {MAX_TOOL_ROUNDS}")
        # a. Append assistant message *before* any tool calls
        msgs.append(res_msg)

        for tc in res_msg.tool_calls:
            # b. Call the respective tool and append tool results
            msgs.append(
                {
                    "role": "tool",
                    "content": dispatch_tool(
                        tc.function.name, json.loads(tc.function.arguments) or {}
                    ),
                    "tool_call_id": tc.id,
                }
            )

        # c. Call the LLM with updated msgs
        res = _client.chat.completions.create(
            model=LLM_MODEL,
            messages=msgs,
            tools=TOOL_DEFINITIONS,
            tool_choice="auto",  # default
        )

    # 4. Return the final text response
    fallback = (
        ""
        if i <= MAX_TOOL_ROUNDS
        else "Thought process interrupted - answer may not be complete.\n\n"
    )
    return fallback + (res_msg.content or "")
