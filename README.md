# Snapdeal Virtual KAM (mini voice agent)

Browser voice bot that acts as a Key Account Manager for a seller whose sales dropped.

## Run
1. Get a free key at https://aistudio.google.com/apikey
2. `cp .env.example .env` and paste the key
3. `python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`
4. `uvicorn app.main:app --reload`
5. Open http://localhost:8000 in **Chrome**, click **Start call**, allow the mic.

## Try saying
"Haan, meri sales bahut gir gayi hai, samajh nahi aa raha kyun" (switch language to Hindi)
or "My sales dropped a lot this month, can you help me?"

## Architecture
Mic -> Web Speech API (STT) -> FastAPI POST /api/chat -> LangGraph agent -> reply text -> browser TTS.

LangGraph: START -> agent (Gemini via langchain-google-genai, tools bound) -> tools_condition
  -> tools (ToolNode: get_performance_summary, diagnose_drop, save_action_plan) -> agent -> ... -> END.
Conversation memory is a MemorySaver checkpointer keyed by the call's session_id (thread_id).
Numbers come only from tool output. Maths is done in code, not by the LLM.

## Assumptions
- Seller name "Rakesh" is made up. "Orders" = orders placed; cancellations come out of that.
- No competitor, traffic or rating data exists in the brief, so visibility is a hypothesis the bot confirms by asking.
- Action plan "WhatsApp + callback" is simulated (logged and shown in the UI).
