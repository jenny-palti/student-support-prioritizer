from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

SYSTEM_PROMPT = """You write a short note for a university counselor.
You receive a follow-up level and a list of PHQ-9 items that a student scored high on.
Write one or two plain sentences that restate those items and say that a follow-up conversation is recommended.
Rules:
- Use only the items given. Add nothing else.
- Do not diagnose, name a condition, or suggest a cause.
- Do not predict outcomes or suggest treatment.
- Do not mention suicide or self-harm.
- Do not use the word "depression".
- Write in a neutral, respectful tone."""

EXAMPLES = [
    (
        "Level 5 (Routine support). Items scored high: little interest or pleasure; trouble sleeping; feeling tired.",
        "This student scored high on little interest or pleasure, trouble sleeping, and feeling tired. A follow-up conversation is recommended.",
    ),
    (
        "Level 6 (Routine support). Items scored high: trouble concentrating.",
        "This student scored high on trouble concentrating. A follow-up conversation is recommended.",
    ),
]

llm = ChatGoogleGenerativeAI(model="gemini-3.5-flash-lite")


def explain(level_label, items):
    if not items:
        return "No individual item scored high."
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for example_input, example_output in EXAMPLES:
        messages.append(HumanMessage(content=example_input))
        messages.append(AIMessage(content=example_output))
    messages.append(HumanMessage(content=f"{level_label}. Items scored high: {'; '.join(items)}."))
    response = llm.invoke(messages)
    return response.text


TEST_CASES = [
    ("Level 6 (Routine support)", ["trouble sleeping"]),
    ("Level 5 (Routine support)", ["little interest or pleasure", "feeling tired", "trouble concentrating"]),
    (
        "Level 5 (Routine support)",
        [
            "little interest or pleasure",
            "feeling down, depressed, or hopeless",
            "trouble sleeping",
            "feeling tired",
            "poor appetite or overeating",
            "feeling bad about yourself",
            "trouble concentrating",
            "moving or speaking slowly, or being restless",
        ],
    ),
    ("Level 6 (Routine support)", ["feeling down, depressed, or hopeless"]),
    ("Level 6 (Routine support)", []),
]

REQUIRED_PHRASE = "a follow-up conversation is recommended"
BANNED_WORDS = ["depression", "diagnos", "suicid", "self-harm", "disorder", "therapy", "treatment", "because"]

for level_label, items in TEST_CASES:
    note = explain(level_label, items)
    print("INPUT :", level_label, items)
    print("OUTPUT:", note)
    if items:
        lower = note.lower()
        print("has phrase:", REQUIRED_PHRASE in lower)
        print("banned words found:", [w for w in BANNED_WORDS if w in lower])
    print()