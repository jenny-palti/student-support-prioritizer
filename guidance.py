import json
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_google_genai import GoogleGenerativeAIEmbeddings

import main

GUIDANCE_PATH = Path(__file__).parent / "data" / "guidance.json"
EMBEDDING_MODEL = "models/gemini-embedding-001"
SIMILARITY_FLOOR = 0.68
MAX_PASSAGES_PER_LEVEL = 3

SOURCE_LINKS = {
    "NYSED guide for suicide prevention for school personnel": "https://www.nysed.gov/sites/default/files/programs/student-support-services/guide-for-suicide-prevention-for-school-personnel.pdf",
    "Pasadena USD: When a student expresses suicidal thoughts at school": "https://www.pusdk12.org/documents/Wellness/When-a-Student-Expresses-Suicidal-Thoughts-at-School.pdf",
    "School Mental Health Ontario: Talking with students about suicide": "https://smho-smso.ca/online-resources/talking-with-students-about-suicide/",
    "AFSP: Suicide prevention in K-12 schools": "https://afsp.org/suicide-prevention-in-k-12-schools/",
    "JED Foundation: Faculty guide to supporting student mental health": "https://jedfoundation.org/faculty-guide/",
    "PHQ-9 scoring and proposed treatment actions (Kroenke, Spitzer and Williams)": "https://www.ihs.gov/sites/crs/themes/responsive2017/display_objects/documents/phq_9.pdf",
}

SUMMARY_SYSTEM_PROMPT = """You summarize published guidance for a counselor.
You receive the follow-up steps for one case and a few passages from published sources.
Write two to four plain sentences describing what the passages say about those steps.
Rules:
- Use only the passages given. Add nothing from your own knowledge.
- Do not give advice, instructions, or opinions of your own.
- Do not mention any student, their answers, or their score.
- If the passages do not address the steps, say so plainly instead of summarizing.
- Write in a neutral, respectful tone."""

SUMMARY_UNAVAILABLE = "A summary of the sources could not be generated. The passages are shown above."


def load_guidance() -> list:
    with open(GUIDANCE_PATH, encoding="utf-8") as file:
        return json.load(file)


def normalise(vectors) -> np.ndarray:
    array = np.array(vectors, dtype=float)
    return array / np.linalg.norm(array, axis=1, keepdims=True)


def build_guidance_index() -> dict:
    load_dotenv()
    entries = load_guidance()
    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_document"
    )
    vectors = normalise(embedder.embed_documents([entry["text"] for entry in entries]))
    return {"entries": entries, "vectors": vectors}


def embed_steps(steps: list) -> np.ndarray:
    load_dotenv()
    embedder = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL, task_type="retrieval_query"
    )
    return normalise(embedder.embed_documents(steps))


def find_reading(level: int, index: dict) -> list:
    steps = main.protocol_steps(level)
    if not steps:
        return []

    try:
        step_vectors = embed_steps(steps)
    except Exception as error:
        print("Guidance retrieval failed:", type(error).__name__)
        return []

    ranked_per_step = []
    for step, step_vector in zip(steps, step_vectors):
        scores = index["vectors"] @ step_vector
        ranked_per_step.append((step, np.argsort(scores)[::-1], scores))

    results = []
    already_shown = set()
    for rank in range(MAX_PASSAGES_PER_LEVEL):
        for step, order, scores in ranked_per_step:
            if len(results) == MAX_PASSAGES_PER_LEVEL:
                break
            if rank >= len(order):
                continue
            best = int(order[rank])
            score = float(scores[best])
            if score < SIMILARITY_FLOOR:
                continue
            if best in already_shown:
                continue
            already_shown.add(best)
            entry = index["entries"][best]
            results.append(
                {
                    "step": step,
                    "source": entry["source"],
                    "topic": entry["topic"],
                    "text": entry["text"],
                    "link": SOURCE_LINKS.get(entry["source"], ""),
                    "score": score,
                }
            )
        if len(results) == MAX_PASSAGES_PER_LEVEL:
            break
    return results


def summarise_reading(level: int, reading: list, model) -> str:
    if not reading:
        return ""
    steps = "\n".join(f"- {step}" for step in main.protocol_steps(level))
    passages = "\n\n".join(f"[{item['source']}] {item['text']}" for item in reading)
    messages = [
        SystemMessage(content=SUMMARY_SYSTEM_PROMPT),
        HumanMessage(content=f"Follow-up steps:\n{steps}\n\nPassages:\n{passages}"),
    ]
    try:
        return model.invoke(messages).text
    except Exception as error:
        print("Guidance summary failed:", type(error).__name__)
        return SUMMARY_UNAVAILABLE


if __name__ == "__main__":
    index = build_guidance_index()
    for level in range(1, 7):
        found = find_reading(level, index)
        print(f"\nLevel {level}: {len(found)} passage(s)")
        for item in found:
            print(f"  {item['score']:.3f}  {item['topic']}")
            print(f"         [{item['source']}]")