import os
import streamlit as st

import main

try:
    os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
except Exception:
    pass

@st.cache_resource
def get_explainer_model():
    return main.build_explainer_model()


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
    st.write("The pre-screen will go here.")