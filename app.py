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

ANSWER_OPTIONS = {
    0: "Not at all",
    1: "Several days",
    2: "More than half the days",
    3: "Nearly every day",
}


def make_example(stress, anxiety):
    values = {}
    for (key, _), value in zip(STRESS_ITEMS, stress):
        values[key] = value
    for (key, _), value in zip(ANXIETY_ITEMS, anxiety):
        values[key] = value
    return values


EXAMPLE_LOW = make_example([2, 2, 2, 3, 3, 2, 3, 3, 2, 2], [1, 1, 1, 1, 1, 1, 0])
EXAMPLE_MIDDLE = make_example([3, 3, 3, 2, 2, 3, 2, 2, 3, 3], [1, 1, 1, 1, 1, 1, 1])
EXAMPLE_HIGH = make_example([3, 3, 4, 2, 2, 3, 2, 1, 3, 3], [2, 2, 2, 2, 2, 2, 2])


def load_example(values):
    for key, value in values.items():
        st.session_state[key] = value


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
    st.subheader("Pre-screen: in which wave should this student be assessed?")
    st.caption(
        "This tool does not detect depression and does not decide who is assessed. "
        "Every student is offered the PHQ-9. The pre-screen suggests the order. "
        "The counselor decides."
    )

    st.markdown("**Try an example** (typical answer patterns, not individual students)")
    example_columns = st.columns(3)
    example_columns[0].button("Typical wave 3", on_click=load_example, args=(EXAMPLE_LOW,))
    example_columns[1].button("Typical wave 2", on_click=load_example, args=(EXAMPLE_MIDDLE,))
    example_columns[2].button("Typical wave 1", on_click=load_example, args=(EXAMPLE_HIGH,))
    prescreen_values = {}

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
    wave = main.assign_wave(probability)
    wave_info = main.WAVES[wave]

    st.subheader("Suggested priority")
    headline = wave_info["label"]
    if wave == 1:
        st.error(headline)
    elif wave == 2:
        st.warning(headline)
    else:
        st.info(headline)

    st.write(f"**Screening:** {wave_info['screening']}")
    st.write(f"**Session:** {wave_info['session']}")

    st.caption(
        "A later wave does not mean no risk. If a student mentions self-harm, do not "
        "wait for this tool: follow the crisis protocol. The counselor decides."
    )

    st.caption(
        "Next step: the PHQ-9 is completed in the first tab. That assigns the follow-up "
        "level, from 1 to 6, and applies the crisis rule to item 9."
    )

    with st.expander("The three waves"):
        for number, info in main.WAVES.items():
            st.write(f"**{info['label']}**")
            st.write(f"Screening: {info['screening']}")
            st.write(f"Session: {info['session']}")
            st.write("")

    with st.expander("The six follow-up levels (after the PHQ-9)"):
        for number, info in main.LEVEL_INFO.items():
            st.write(f"Level {number}: {info['label']} (Protocol {info['protocol']})")
