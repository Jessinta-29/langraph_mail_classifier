# run.py
from graph import app

email = """
Hi Jess, could you send the final slides for tomorrow's client meeting?
Best,
Michael
"""

input_state = {
    "email": email,
    "category": "unknown",
    "response": ""
}

result = app.invoke(input_state)

print(" Category:", result["category"])
print(" Response:\n", result["response"])
