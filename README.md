# Plant Advisor — AI201 Lab 2

**_CodePath AI201: Applications of AI Engineering (Summer 2026)_**

_Krish A. Patel_

A conversational agent that helps users care for their houseplants. Ask it anything about a plant in its database and it will look up the care requirements, check the current seasonal context, and give you specific, grounded advice.

The app is built and running. One of the tools and the agent weren't functional to begin with, so I implemented them for this lab in milestones.

---

## Demo

Query:

> How should I care for my pothos in this season?

Response:

> According to the care data for your pothos, it should be watered every 1–2 weeks, allowing the top inch of soil to dry out between waterings. As we are currently in summer, you should water your pothos more frequently, checking the soil every few days, and continue regular fertilizing. Be careful about direct afternoon sun through windows, which can be intense enough to burn leaves. Also, watch out for pests like spider mites and fungus gnats, which thrive in hot, dry conditions and can appear with overwatering. Keep an eye on your plant for signs of heat stress or air conditioning stress.

Calls the tools `lookup_plant({'plant_name': 'pothos'})` and `get_seasonal_conditions({})`, which auto-detects the season as summer since the query was run in June.

---

## Key Design Choices

- Upon hitting edge cases (discussed in its own section below), the agent doesn't just return an unhelpful and static error string - rather, it is allowed to pull from its general (training) knowledge as long as it acknowledges that it isn't in the database. This balances usefulness with accuracy, ensuring a good user experience.
- The agent loop has a hard cap of `MAX_TOOL_ROUNDS` to prevent runaway tool calls. If this limit is hit without a natural end to tool calling by the LLM, the end response is modified to reflect that.
- When using the tool (function) `lookup_plants` to fetch plant-specific info from the `plants.json` database, the search is performed in this order:
  1. Check if given name is a key in the dictionary - reason: first check key membership for potential `O(1)` answers
  2. Check if given name is a display name of that plant - reason: display names may be slightly different than the key slugs (in more ways than just case and end whitespace); run 2nd since each plant has exactly one display name
  3. Check if given name is an alias - reason: users may know the plant by another term; save for last since it's the broadest search (potentially multiple aliases per plant)
- The tool `get_seasonal_conditions` (which fetches general plant conditions per season from the `seasons.json` database) optionally accepts a valid season parameter, but uses the user's current season as fallback (via `datetime`) to satisfy queries that indirectly mention season (e.g. "this season", "right around now", etc.).
- `400`-level errors from LLM calls are gracefully handled in multiple layers. The errors occur when the LLM rejects making a tool call. The first fallback retries the LLM call without tool use to salvage the user experience by providing a response solely from the LLM's training data. Only when that fails does the agent return a generic error string.

---

## Setup

**1. Create and activate a virtual environment:**

```bash
python -m venv .venv
source .venv/bin/activate      # Mac/Linux
# or: .venv\Scripts\activate   # Windows
```

**3. Install dependencies:**

```bash
pip install -r requirements.txt
```

**4. Add your Groq API key.** Copy `.env.example` to `.env` and paste in your key from [console.groq.com](https://console.groq.com).

**5. Run the app:**

```bash
python app.py
```

Plant Advisor will open in your browser.

---

## Project Structure

```
ai201-lab2-plantadvisor-starter/
├── app.py              ← Gradio UI (complete)
├── config.py           ← API keys and settings (complete)
├── agent.py            ← Tool definitions + run_agent() - Milestone 2 implemention
├── tools.py            ← lookup_plant() and get_seasonal_conditions() to Milestone 1 implemention
├── data/
│   ├── plants.json     ← 15-plant database (complete)
│   └── seasons.json    ← Seasonal care data (complete)
├── specs/
│   ├── system-design.md        ← Start here
│   ├── tool-functions-spec.md  ← Completed before Milestone 1
│   └── agent-loop-spec.md      ← Completed before Milestone 2
└── requirements.txt
```

---

## Edge Case Testing

### Plant Not in Database

Query:

> How do I care for my string of pearls?

Response:

> It seems that I don't have specific information on a plant called "string of pearls" in my database. However, I can offer some general guidance.
> [...]
> Keep in mind that these are general guidelines. If you have more specific questions or concerns about your string of pearls, I'll do my best to help.

Tool call logs:

```
  → Tool call: lookup_plant({'plant_name': 'string of pearls'})
  ← Result: {"found": false, "name": "string of pearls", "message": "Plant 'string of pearls' not found in the database as a key, di...
```

### Irrelevant / Out-of-Domain Query

Query:

> How do I take care of my rabbit?

Response:

> I'm a plant care advisor, and I don't have information about caring for rabbits. However, I can tell you that rabbits are living creatures that require specialized care, including a balanced diet, proper housing, and regular veterinary check-ups. If you're looking for advice on caring for your rabbit, I recommend consulting a veterinarian or a reputable animal care resource.

Tool call logs:

```
[no tool calls]
```
