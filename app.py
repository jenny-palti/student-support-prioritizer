import os
import pandas as pd
import streamlit as st
import main

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

@st.cache_resource
def get_explainer_model():
    return main.build_explainer_model()

@st.cache_resource
def get_prescreen_bundle():
    return main.load_model()

STRESS_OPTIONS = {
    0: "Never",
    1: "Almost never",
    2: "Sometimes",
    3: "Fairly often",
    4: "Very often",
}

SATISFACTION_ITEM = (
    "life_satisfaction",
    "In general, how satisfied are you with all aspects of your life?",
)

YESTERDAY_ITEMS = [
    ("happy_yesterday", "How happy did you feel yesterday?"),
    ("laugh_yesterday", "How much did you laugh yesterday?"),
    ("learn_yesterday", "Did you learn new or exciting things yesterday?"),
    ("enjoy_yesterday", "How much did you enjoy the activities you did yesterday?"),
    ("worried_yesterday", "How worried did you feel yesterday?"),
    ("angry_yesterday", "How angry did you feel yesterday?"),
    ("stress_yesterday", "How much stress did you feel yesterday?"),
    ("lonely_yesterday", "How lonely or unsupported did you feel yesterday?"),
]

STRESS_ITEMS = [
    ("pss1", "been upset because of something that happened unexpectedly"),
    ("pss2", "felt that you were unable to control the important things in your life"),
    ("pss3", "felt nervous and stressed"),
    ("pss4", "felt confident about your ability to handle your personal problems"),
    ("pss5", "felt that things were going your way"),
    ("pss6", "found that you could not cope with all the things that you had to do"),
    ("pss7", "been able to control irritations in your life"),
    ("pss8", "felt that you were on top of things"),
    ("pss9", "been angered because of things that happened that were outside of your control"),
    ("pss10", "felt difficulties were piling up so high that you could not overcome them"),
]

ANXIETY_ITEMS = [
    ("gad1", "Feeling nervous, anxious, or on edge"),
    ("gad2", "Not being able to stop or control worrying"),
    ("gad3", "Worrying too much about different things"),
    ("gad4", "Trouble relaxing"),
    ("gad5", "Being so restless that it is hard to sit still"),
    ("gad6", "Becoming easily annoyed or irritable"),
    ("gad7", "Feeling afraid, as if something awful might happen"),
]

def make_example(life, yesterday, stress, anxiety):
    values = {"life_satisfaction": life}
    for (key, _), value in zip(YESTERDAY_ITEMS, yesterday):
        values[key] = value
    for (key, _), value in zip(STRESS_ITEMS, stress):
        values[key] = value
    for (key, _), value in zip(ANXIETY_ITEMS, anxiety):
        values[key] = value
    return values


EXAMPLE_LOW = make_example(8, [8, 7, 6, 8, 2, 1, 2, 1], [1, 1, 1, 3, 3, 1, 3, 3, 1, 1], [0, 0, 0, 0, 0, 0, 0])
EXAMPLE_MIDDLE = make_example(5, [5, 4, 4, 5, 6, 4, 6, 5], [2, 2, 3, 2, 2, 2, 2, 2, 2, 2], [1, 1, 2, 1, 1, 1, 0])
EXAMPLE_HIGH = make_example(2, [2, 1, 1, 2, 9, 6, 9, 8], [3, 4, 4, 1, 1, 4, 1, 1, 3, 4], [3, 3, 3, 2, 2, 2, 3])


def load_example(values):
    for key, value in values.items():
        st.session_state[key] = value


ANSWER_OPTIONS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day",
}

st.set_page_config(page_title="Student Support Prioritizer")
st.title("Student Support Prioritizer")
st.warning(
    "Do not enter names or any identifying details. "
    "This tool supports a counselor and does not replace one. The counselor decides."
)

phq_tab, prescreen_tab = st.tabs(["PHQ-9 follow-up", "Pre-screen"])

with phq_tab:
    st.subheader("PHQ-9 answers")
    st.caption("Over the last 2 weeks, how often has the student been bothered by each problem?")
    answers = {}
    for i in range(1, 10):
        answers[i] = st.radio(
            f"{i}. {main.PHQ_NAMES[i]}",
            options=list(ANSWER_OPTIONS.keys()),
            format_func=lambda value: ANSWER_OPTIONS[value],
            horizontal=True,
            key=f"phq{i}",
        )

    row = {f"phq{i}": answers[i] for i in range(1, 10)}
    total = sum(answers.values())
    level = main.assign_level(answers[9], total)
    reasons = main.get_reasons(row)
    level_info = main.LEVEL_INFO[level]
    protocol = main.PROTOCOLS[level_info["protocol"]]

    st.divider()
    st.subheader("Follow-up")
    st.write(f"Level {level} of 6: {level_info['label']}")
    st.write(f"Protocol {level_info['protocol']}: {protocol['name']}")

    if protocol["crisis"]:
        st.error("No score is shown. Follow the protocol below.")
    else:
        st.write(f"PHQ-9 total: {total} of 27 ({main.severity_band(total)})")
    st.markdown(protocol["text"].replace("\n", "\n\n"))

    st.write("Items that scored high:")
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.write("No individual item scored high.")

    if not protocol["crisis"] and reasons:
        if st.button("Summarize the high-scoring questions"):
            with st.spinner("Writing the note..."):
                note = main.explain_reasons(level, reasons, get_explainer_model())
            if note:
                st.info(note)    

with prescreen_tab:
    st.subheader("Pre-screen: who should be invited to complete the PHQ-9?")
    st.caption(
        "This tool does not detect depression. It suggests who to invite for the "
        "PHQ-9. The counselor decides."
    )

    st.markdown("**Try an example** (made-up answers, not real students)")
    example_columns = st.columns(3)
    example_columns[0].button("Low distress", on_click=load_example, args=(EXAMPLE_LOW,))
    example_columns[1].button("In between", on_click=load_example, args=(EXAMPLE_MIDDLE,))
    example_columns[2].button("High distress", on_click=load_example, args=(EXAMPLE_HIGH,))
    prescreen_values = {}

    st.markdown("**Overall**")
    key, text = SATISFACTION_ITEM
    prescreen_values[key] = st.slider(
        text + " (0 = not satisfied, 10 = totally satisfied)", 0, 10, 5, key=key
    )

    st.markdown("**Yesterday** (0 = at no time, 10 = all the time)")
    for key, text in YESTERDAY_ITEMS:
        prescreen_values[key] = st.slider(text, 0, 10, 5, key=key)

    st.markdown("**Stress in the last month**")
    st.caption("In the last month, how often have you...")
    for key, text in STRESS_ITEMS:
        prescreen_values[key] = st.radio(
            text,
            options=list(STRESS_OPTIONS.keys()),
            format_func=lambda value: STRESS_OPTIONS[value],
            horizontal=True,
            key=key,
        )

    st.markdown("**Worry and tension in the last two weeks**")
    st.caption("Over the last two weeks, how often have you been bothered by...")
    for key, text in ANXIETY_ITEMS:
        prescreen_values[key] = st.radio(
            text,
            options=list(ANSWER_OPTIONS.keys()),
            format_func=lambda value: ANSWER_OPTIONS[value],
            horizontal=True,
            key=key,
        )
    st.divider()
    bundle = get_prescreen_bundle()
    answers_table = pd.DataFrame([prescreen_values])[bundle["features"]]
    probability = bundle["model"].predict_proba(answers_table)[0, 1]

    st.subheader("Suggestion")
    if probability >= bundle["cutoff"]:
        st.warning("The pre-screen suggests inviting this student to complete the PHQ-9.")
    else:
        st.info("The pre-screen does not suggest an invitation for this student.")
    st.caption(
        "Not suggested does not mean no risk. If a student mentions self-harm, do not "
        "wait for this tool: follow the crisis protocol. The counselor decides."
    )

    st.caption(
        "Next step: if the student is invited, complete the PHQ-9 in the first tab. "
        "That assigns the follow-up level, from 1 to 6."
    )
    with st.expander("The six follow-up levels"):
        for number, info in main.LEVEL_INFO.items():
            st.write(f"Level {number}: {info['label']} (Protocol {info['protocol']})")