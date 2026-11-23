import os
from concurrent.futures import ThreadPoolExecutor

import pandas as pd
import streamlit as st
import main
import guidance

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

@st.cache_resource
def get_guidance_index():
    try:
        return guidance.build_guidance_index()
    except Exception as error:
        print("Guidance index failed:", type(error).__name__)
        return None

@st.cache_data(show_spinner=False)
def get_reading(level):
    index = get_guidance_index()
    if index is None:
        return []
    return guidance.find_reading(level, index)

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

BAND_COLORS = {
    "Minimal": "blue",
    "Mild": "blue",
    "Moderate": "orange",
    "Moderately severe": "red",
    "Severe": "red",
}

ITEM9_COLORS = {0: "blue", 1: "orange", 2: "red", 3: "red"}

CLAMP_STYLE = (
    "display:-webkit-box;-webkit-line-clamp:2;-webkit-box-orient:vertical;overflow:hidden"
)


def build_case_recap(item9, total, band, level):
    info = main.LEVEL_INFO[level]
    protocol = main.PROTOCOLS[info["protocol"]]
    step_count = len(main.protocol_steps(level))
    plural = "s" if step_count != 1 else ""
    return (
        f"Item 9: {ANSWER_OPTIONS[item9]} ({item9}). "
        f"PHQ-9 total {total} of 27, {band.lower()}. "
        f"Level {level}, Protocol {info['protocol']} "
        f"({protocol['name'].lower()}) — {info['label'].lower()}. "
        f"{step_count} protocol step{plural} apply."
    )


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


def make_phq_example(values):
    return {f"phq{i}": value for i, value in enumerate(values, start=1)}


PHQ_EXAMPLES = {
    1: make_phq_example([3, 3, 3, 3, 2, 2, 1, 1, 3]),
    2: make_phq_example([2, 2, 2, 1, 1, 1, 1, 0, 2]),
    3: make_phq_example([3, 3, 2, 2, 2, 2, 1, 1, 1]),
    4: make_phq_example([2, 2, 1, 1, 1, 0, 0, 0, 1]),
    5: make_phq_example([3, 3, 3, 3, 3, 3, 2, 1, 0]),
    6: make_phq_example([2, 1, 0, 0, 0, 0, 0, 0, 0]),
}


def load_example(values):
    for key, value in values.items():
        st.session_state[key] = value


st.set_page_config(page_title="Student Support Prioritizer")

st.markdown(
    """
    <style>
    html { font-size: 18px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Student Support Prioritizer")

prescreen_tab, phq_tab = st.tabs(["Pre-screen", "PHQ-9 follow-up"])

with phq_tab:
    st.subheader("Patient health questionnaire-9 (PHQ-9)")
    st.caption(
        "This tool is a standard screening questionnaire which records how often a "
        "student has experienced nine depressive symptoms over the last two weeks."
    )

    with st.container(border=True):
        st.markdown("**Try an example** (typical answer patterns, not individual students)")
        first_row = st.columns(3)
        second_row = st.columns(3)
        for position, example_level in enumerate([1, 2, 3]):
            first_row[position].button(
                f"Typical level {example_level}",
                on_click=load_example,
                args=(PHQ_EXAMPLES[example_level],),
                key=f"phq_example_{example_level}",
            )
        for position, example_level in enumerate([4, 5, 6]):
            second_row[position].button(
                f"Typical level {example_level}",
                on_click=load_example,
                args=(PHQ_EXAMPLES[example_level],),
                key=f"phq_example_{example_level}",
            )

    st.markdown("Over the last 2 weeks, how often have you been bothered by each problem?")

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

    item9 = answers[9]
    band = main.severity_band(total)

    st.markdown(
        f"The student answered :{ITEM9_COLORS[item9]}[**{ANSWER_OPTIONS[item9]} ({item9})**] "
        f"on PHQ-9 item 9 — {main.PHQ_NAMES[9].lower()}."
    )
    st.markdown(f"PHQ-9 total: {total} of 27 — :{BAND_COLORS[band]}[**{band}**]")

    st.markdown(
        f"**Protocol {level_info['protocol']} / Level {level}: "
        f"{protocol['name']} / {level_info['label']}**"
    )

    for number, step in enumerate(main.protocol_steps(level), start=1):
        st.markdown(f"{number}. {step}")

    st.markdown("**Items that scored high:**")
    if reasons:
        for reason in reasons:
            st.markdown(f"- {reason}")
    else:
        st.write("No individual item scored high.")

    reading = get_reading(level)
    if reading:
        st.markdown("**Suggested further reading and resources**")
        for item in reading:
            with st.container(border=True):
                st.markdown(f"**{item['topic'][0].upper()}{item['topic'][1:]}**")
                st.markdown(
                    f'<div style="{CLAMP_STYLE}">{item["text"]}</div>',
                    unsafe_allow_html=True,
                )
                st.caption(f"Source: {item['source']}")
                if item["link"]:
                    st.link_button("Read more", item["link"])

    if st.button("Generate summary"):
        with st.spinner("Writing the summary..."):
            model = get_explainer_model()
            with ThreadPoolExecutor(max_workers=2) as pool:
                student_job = (
                    pool.submit(main.explain_reasons, level, reasons, model) if reasons else None
                )
                sources_job = pool.submit(guidance.summarise_reading, level, reading, model)
                student_note = student_job.result() if student_job else ""
                sources_note = sources_job.result()
        recap = build_case_recap(item9, total, band, level)
        st.markdown("**This student's answers**")
        st.info(f"{recap}\n\n{student_note}".strip())
        if sources_note:
            st.markdown("**What the sources say**")
            st.info(sources_note)

    with st.expander("The six follow-up levels"):
        for number, info in main.LEVEL_INFO.items():
            st.write(f"Protocol {info['protocol']} / Level {number}: {info['label']}")

with prescreen_tab:
    st.subheader("Pre-screen: in which wave should this student be assessed?")
    st.caption(
        "This tool does not detect depression and does not decide who is assessed. "
        "Every student is offered the PHQ-9. The pre-screen suggests the order."
    )

    with st.container(border=True):
        st.markdown("**Try an example** (typical answer patterns, not individual students)")
        example_columns = st.columns(3)
        example_columns[0].button("Typical wave 1", on_click=load_example, args=(EXAMPLE_HIGH,))
        example_columns[1].button("Typical wave 2", on_click=load_example, args=(EXAMPLE_MIDDLE,))
        example_columns[2].button("Typical wave 3", on_click=load_example, args=(EXAMPLE_LOW,))

    prescreen_values = {}

    st.markdown("**Stress in the last month**")
    st.markdown("In the last month, how often have you...")
    for key, text in STRESS_ITEMS:
        prescreen_values[key] = st.radio(
            text,
            options=list(STRESS_OPTIONS.keys()),
            format_func=lambda value: STRESS_OPTIONS[value],
            horizontal=True,
            key=key,
        )

    st.markdown("**Worry and tension in the last two weeks**")
    st.markdown("Over the last two weeks, how often have you been bothered by...")
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

    with st.expander("The three waves"):
        for number, info in main.WAVES.items():
            st.write(f"**{info['label']}**")
            st.write(f"Screening: {info['screening']}")
            st.write(f"Session: {info['session']}")
            st.write("")