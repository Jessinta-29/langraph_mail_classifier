# graph.py
import os
from dotenv import load_dotenv
from typing import TypedDict, Literal
from langgraph.graph import StateGraph, START, END
from llama_index.llms.groq import Groq

# === Load API Key ===
load_dotenv()
llm = Groq(
    model="deepseek-r1-distill-llama-70b",  # or "mixtral-8x7b"
    api_key=os.getenv("GROQ_API_KEY")
)

# === Define State ===
class State(TypedDict):
    email: str
    category: Literal["work", "personal", "spam", "unknown"]
    response: str

# === Node 1: Classify Email ===
def classify(state: State) -> State:
    prompt = f"Classify this email as one of: work, personal, spam, unknown:\n{state['email']}"
    result = llm.complete(prompt)
    return {**state, "category": result.text.strip().lower()}

# === Node 2: Generate Response ===
def respond(state: State) -> State:
    category = state["category"]
    if "work" in category or "personal" in category:
        prompt = f"""
You are an assistant helping reply to emails.

This is a {category} email:
\"\"\"
{state['email']}
\"\"\"

Reply with a short, polite message (1–2 sentences) that acknowledges and answers the email appropriately.
"""
        result = llm.complete(prompt)
        return {**state, "response": result.text.strip()}
    return {**state, "response": "No reply needed."}

# === Build Graph ===
graph = StateGraph(State)
graph.add_node("classify", classify)
graph.add_node("respond", respond)
graph.set_entry_point("classify")
graph.add_edge("classify", "respond")
graph.add_edge("respond", END)

# === Compile Graph ===
app = graph.compile()
