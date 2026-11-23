import joblib
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI


PHQ_ITEM_COLUMNS = [f"phq{i}" for i in range(1, 10)]
GAD_ITEM_COLUMNS = [f"gad{i}" for i in range(1, 8)]
PSS_ITEM_COLUMNS = [f"pss{i}" for i in range(1, 11)]
PSS_REVERSED_ITEMS = [4, 5, 7, 8]

PHQ_NAMES = {
    1: "Little interest or pleasure in doing things",
    2: "Feeling down, depressed or hopeless",
    3: "Trouble with sleep",
    4: "Feeling tired or having little energy",
    5: "Poor appetite or overeating",
    6: "Feeling bad about yourself or like a failure",
    7: "Trouble concentrating",
    8: "Moving or speaking very slowly, or being restless",
    9: "Thoughts of being better off dead or of self-harm",
}

PROTOCOLS = {
    "A": {
        "name": "Immediate safety follow-up",
        "crisis": True,
        "steps": [
            "A trained counselor carries out a structured suicide risk assessment (for example the C-SSRS) and, where possible, consults a second trained colleague before deciding on the level of care.",
            "If the student shows immediate danger (a plan, intent, or access to means): do not leave them alone, remove means if it is safe to do so, and call emergency services (your local emergency number).",
            "Contact the parent or guardian according to institutional policy, and advise securing medications and other dangerous items.",
            "Complete a safety plan with the student: warning signs, coping strategies, supports, crisis line numbers.",
            "Refer to community mental health services with 24-hour access if needed, and obtain consent to share information with the providers.",
            "Document every contact and decision. Follow up within one week and again at one month.",
        ],
    },
    "B": {
        "name": "Clinician review",
        "crisis": True,
        "steps": [
            "In the conversation, ask directly about suicidal thoughts and carry out a structured screening (for example the C-SSRS screener).",
            "If the screening suggests higher risk, move to Protocol A at once.",
            "If not, agree a safety plan where needed, share resources and counseling options, and involve the parent or guardian according to policy.",
            "Document the contact and the plan. Check in again within 2 weeks.",
        ],
    },
    "C": {
        "name": "Support and check-in",
        "crisis": False,
        "steps": [
            {
                5: "Offer a check-in conversation and information about support services. A high total without a crisis answer still calls for a timely, scheduled check-in. This tool only sorts the order.",
                6: "Standard support. Share wellbeing resources.",
            },
        ],
    },
}

LEVEL_INFO = {
    1: {"protocol": "A", "label": "Contact the same day"},
    2: {"protocol": "A", "label": "Contact within 24-48 hours"},
    3: {"protocol": "B", "label": "Review and contact within 3 days"},
    4: {"protocol": "B", "label": "Review and contact within 4 days"},
    5: {"protocol": "C", "label": "Offer check-in"},
    6: {"protocol": "C", "label": "Standard"},
}


EXPLAINER_MODEL = "gemini-3.5-flash-lite"

EXPLAINER_TIMEOUT_SECONDS = 30

SHORT_NAMES = {
    PHQ_NAMES[2]: "feeling down or hopeless",
    PHQ_NAMES[8]: "moving or speaking very slowly or being restless",
}

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


NOTE_UNAVAILABLE = "A note could not be generated. The items scored high are listed below."

ITEM9_SENTENCE = "The student also answered above 0 on PHQ-9 item 9 (thoughts of being better off dead or of self-harm)."


EXAMPLES = [
    (
        "Level 5 (Support and check-in). Items scored high: little interest or pleasure; trouble sleeping; feeling tired.",
        "This student scored high on little interest or pleasure, trouble sleeping, and feeling tired. A follow-up conversation is recommended.",
    ),
    (
        "Level 6 (Support and check-in). Items scored high: trouble concentrating.",
        "This student scored high on trouble concentrating. A follow-up conversation is recommended.",
    ),
]


def severity_band(total: int) -> str:
    if total <= 4:
        return "Minimal"
    if total <= 9:
        return "Mild"
    if total <= 14:
        return "Moderate"
    if total <= 19:
        return "Moderately severe"
    return "Severe"


def gad7_band(total: int) -> str:
    if total <= 4:
        return "minimal range"
    if total <= 9:
        return "mild range"
    if total <= 14:
        return "moderate range"
    return "severe range"


def pss10_band(total: int) -> str:
    if total <= 13:
        return "low range"
    if total <= 26:
        return "moderate range"
    return "high range"


def score_gad7(values: dict) -> dict:
    total = sum(values[f"gad{i}"] for i in range(1, 8))
    return {"total": total, "max": 21, "band": gad7_band(total)}


def score_pss10(values: dict) -> dict:
    total = 0
    for i in range(1, 11):
        answer = values[f"pss{i}"]
        if i in PSS_REVERSED_ITEMS:
            answer = 4 - answer
        total += answer
    return {"total": total, "max": 40, "band": pss10_band(total)}


def assign_level(item9: int, total: int) -> int:
    if item9 == 3:
        return 1
    if item9 == 2:
        return 2
    if item9 == 1 and total >= 15:
        return 3
    if item9 == 1:
        return 4
    if total >= 15:
        return 5
    return 6


def protocol_steps(level: int) -> list:
    protocol = PROTOCOLS[LEVEL_INFO[level]["protocol"]]
    steps = []
    for step in protocol["steps"]:
        if isinstance(step, dict):
            if level in step:
                steps.append(step[level])
        else:
            steps.append(step)
    return steps


def get_reasons(row) -> list:
    reasons = []
    if row["phq9"] >= 1:
        reasons.append(PHQ_NAMES[9])
    for i in range(1, 9):
        if row[f"phq{i}"] >= 2:
            reasons.append(PHQ_NAMES[i])
    return reasons


def build_explainer_model():
    load_dotenv()
    return ChatGoogleGenerativeAI(model=EXPLAINER_MODEL, timeout=EXPLAINER_TIMEOUT_SECONDS)


def explain_reasons(level: int, reasons: list, model) -> str | None:
    protocol = LEVEL_INFO[level]["protocol"]
    item9_flagged = PHQ_NAMES[9] in reasons
    other_reasons = [name for name in reasons if name != PHQ_NAMES[9]]

    if not other_reasons:
        if item9_flagged:
            return ITEM9_SENTENCE
        return "No individual item scored high."

    items = [SHORT_NAMES.get(name, name.lower()) for name in other_reasons]
    level_label = f"Level {level} ({PROTOCOLS[protocol]['name']})"
    messages = [SystemMessage(content=SYSTEM_PROMPT)]
    for example_input, example_output in EXAMPLES:
        messages.append(HumanMessage(content=example_input))
        messages.append(AIMessage(content=example_output))
    messages.append(HumanMessage(content=f"{level_label}. Items scored high: {'; '.join(items)}."))
    try:
        note = model.invoke(messages).text
    except Exception as error:
        print("Explainer failed:", type(error).__name__)
        note = NOTE_UNAVAILABLE

    if item9_flagged:
        return f"{note} {ITEM9_SENTENCE}"
    return note


def add_phq_columns(df: pd.DataFrame) -> pd.DataFrame:
    data = df.copy()
    data["phq_total"] = data[PHQ_ITEM_COLUMNS].sum(axis=1)
    data["phq_band"] = data["phq_total"].apply(severity_band)
    data["phq10_plus"] = (data["phq_total"] >= 10).astype(int)
    data["item9_positive"] = (data["phq9"] >= 1).astype(int)
    data["level"] = [assign_level(a, t) for a, t in zip(data["phq9"], data["phq_total"])]
    data["reasons"] = data.apply(get_reasons, axis=1)
    return data


FEATURE_COLUMNS = PSS_ITEM_COLUMNS + GAD_ITEM_COLUMNS
GROUP_COLUMNS = ["gender", "ses", "ethnicity"]

WAVE_BOUNDARIES = {1: 0.50, 2: 0.30}
PRIORITISE_THRESHOLD = WAVE_BOUNDARIES[2]

WAVES = {
    1: {
        "label": "Wave 1",
        "screening": "Immediately administer PHQ-9.",
        "session": "Immediately invite for a session and administer C-SSRS.",
    },
    2: {
        "label": "Wave 2",
        "screening": "Immediately administer PHQ-9.",
        "session": "Invite for a session and administer C-SSRS after the first wave, guided by the results of those assessments.",
    },
    3: {
        "label": "Wave 3",
        "screening": "PHQ-9 offered as part of standard screening.",
        "session": "Invite for a session depending on the PHQ-9 score or if the student requests one.",
    },
}


def assign_wave(probability: float) -> int:
    if probability >= WAVE_BOUNDARIES[1]:
        return 1
    if probability >= WAVE_BOUNDARIES[2]:
        return 2
    return 3


def prepare_features_and_target(df: pd.DataFrame):
    X = df[FEATURE_COLUMNS]
    y = df["item9_positive"]
    groups = df[GROUP_COLUMNS]
    return X, y, groups


def split_data(X, y, groups):
    return train_test_split(X, y, groups, test_size=0.2, random_state=42, stratify=y)


def build_model() -> Pipeline:
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            ("logistic", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )


def train_model(model, X_train, y_train):
    model.fit(X_train, y_train)
    return model


def evaluate_model(model, X_test, y_test, cutoff: float = PRIORITISE_THRESHOLD):
    probabilities = model.predict_proba(X_test)[:, 1]
    predictions = (probabilities >= cutoff).astype(int)
    print(classification_report(y_test, predictions, target_names=["item 9 = 0", "item 9 above 0"]))
    return probabilities, predictions


def wave_summary(probabilities, y_true) -> None:
    table = pd.DataFrame(
        {"wave": [assign_wave(p) for p in probabilities], "answered_above_0": list(y_true)}
    )
    summary = table.groupby("wave").agg(
        students=("answered_above_0", "size"),
        answered_above_0=("answered_above_0", "sum"),
    )
    summary["share_of_students"] = summary["students"] / len(table)
    summary["rate_within_wave"] = summary["answered_above_0"] / summary["students"]
    print(summary.round(2))
    print()


def save_model(model, path: str = "models/prescreen_model.joblib") -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(
        {"model": model, "wave_boundaries": WAVE_BOUNDARIES, "features": FEATURE_COLUMNS}, path
    )


def load_model(path: str = "models/prescreen_model.joblib") -> dict:
    return joblib.load(path)


def recall_by_group(y_true, pred, groups_df) -> None:
    check = groups_df.copy()
    check["true"] = y_true
    check["invited"] = pred
    check["caught"] = (check["true"] == 1) & (check["invited"] == 1)

    for col in groups_df.columns:
        summary = check.groupby(col).agg(
            students=("true", "size"),
            truly_positive=("true", "sum"),
            caught=("caught", "sum"),
            invited_rate=("invited", "mean"),
        )
        summary["recall"] = summary["caught"] / summary["truly_positive"]
        print(col)
        print(summary.round(2))
        print()


def load_data(file_path: str) -> pd.DataFrame:
    return pd.read_csv(file_path)


def inspect_data(df: pd.DataFrame) -> None:
    print("Shape (rows, columns):", df.shape)
    print("\nColumn types:")
    print(df.dtypes)
    print("\nMissing values per column:")
    print(df.isna().sum())
    print("\nDuplicate rows:", df.duplicated().sum())
    for column in df.select_dtypes(exclude="number").columns:
        print(f"\nValues in {column}:")
        print(df[column].value_counts())
    print("\nSummary of the number columns:")
    print(df.describe().T.to_string())


def main():
    df = add_phq_columns(load_data("data/DATASET-Raw_data_clean.csv"))

    print("Students in the data:", len(df))
    print("Answered above 0 on item 9:", int(df["item9_positive"].sum()))
    print()

    X, y, groups = prepare_features_and_target(df)
    X_train, X_test, y_train, y_test, groups_train, groups_test = split_data(X, y, groups)

    model = train_model(build_model(), X_train, y_train)
    probabilities, predictions = evaluate_model(model, X_test, y_test)

    print("Waves on the test part")
    wave_summary(probabilities, y_test)

    print("Waves 1 and 2 by group (test part)")
    recall_by_group(y_test, predictions, groups_test)

    save_model(model)
    print("Model saved to models/prescreen_model.joblib")


if __name__ == "__main__":
    main()
