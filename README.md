# Student Support Prioritizer

A prioritization aid that helps counselors decide which students to follow up with first, based on screening indicators. **It does not diagnose, and the counselor decides.**

**Live app:** https://student-support-prioritizer.streamlit.app/

Capstone project for the GenAI & Machine Learning Bootcamp (Developers Institute).

## What it does

The tool works in two stages.

**Stage 1: pre-screen.** A logistic regression model reads 26 answers (life satisfaction, feelings yesterday, stress and anxiety questions) and suggests whether the student should be invited to complete the PHQ-9 questionnaire. It shows only a decision, never a probability or a "likelihood of depression".

**Stage 2: PHQ-9 follow-up.** Fixed rules turn the nine PHQ-9 answers into one of six follow-up levels and one of three protocols. Each result lists which items scored high, so the counselor can see why a student was flagged. For the routine protocol only, a Gemini model can write a short plain-language note restating those items.

| Level | Label | PHQ-9 answers | Protocol |
|---|---|---|---|
| 1 | Same day | item 9 = 3 | A: Immediate safety follow-up |
| 2 | Within 24-48 hours | item 9 = 2 | A: Immediate safety follow-up |
| 3 | Priority review | item 9 = 1 and total 15 or higher | B: Clinician review |
| 4 | Scheduled review | item 9 = 1 and total 14 or lower | B: Clinician review |
| 5 | Check-in offered | item 9 = 0 and total 15 or higher | C: Support and check-in |
| 6 | Standard | item 9 = 0 and total 14 or lower | C: Support and check-in |

Item 9 asks about thoughts of being better off dead or of self-harm. The split between levels 3 and 4 is a sorting aid, not a difference in risk.

## Safety design

- **Crisis rule.** If item 9 is above 0, the app shows the escalation protocol and **no score**.
- **No language model near a crisis.** Protocols A and B use fixed text only. The Gemini note exists only for Protocol C, and the code blocks it even there if item 9 appears in the list.
- **The pre-screen cannot see item 9**, so it cannot detect a crisis. "Not suggested" never means "no risk", and the app says so every time.
- The protocol texts were adapted from school suicide-intervention procedures (Mental Health Commission of Canada national guide, NYSED, Pasadena USD, Sherwood SD) and reviewed and finalized by the author, who has a clinical psychology background. The timeframes are the author's own choices.
- The counselor decides. The app asks users not to enter names or identifying details.

## Data

- Source: the 2022 wellbeing survey of the Icesi University community (Colombia). The survey states that its data is anonymous and used for academic purposes. The questionnaire and codebook are described in `data/`.
- The cleaned file has 2,992 students. Sections: demographics, subjective wellbeing (0-10 scales), PSS-10 stress questions (0-4), GAD-7 anxiety questions (0-3), PHQ-9 (0-3).
- Cleaning: 11 impossible birth years and about 120 parent-education values above 25 were noted. No rows were removed. Parent education and birth year are not used as model inputs.
- The project started with a different dataset (Student Depression Dataset) and switched to this survey after instructor feedback, because the survey contains the full PHQ-9 and validated questionnaires.

## Model (stage 1)

- **Target:** a PHQ-9 total of 10 or more (about 40% of students).
- **Inputs (26):** life satisfaction, 8 "yesterday" feelings, PSS-10 and GAD-7 items. All PHQ-9 items, and the "depressed yesterday" question, are excluded because they are too close to the target. Gender, SES and ethnicity are kept out of the inputs and used only afterwards to check results by group.
- **Split:** 80% training (2,393 students), 20% test (599), stratified, `random_state=42`.
- **Models compared:** random forest and logistic regression. In 5-fold cross-validation they were tied (accuracy about 0.82). The random forest overfits (training accuracy 1.00), so the **logistic regression** was chosen: same performance, readable weights.
- **Cut-off 0.35.** An invitation only means completing a questionnaire, so a needless invitation is cheap and a missed student is the costly mistake. A lower cut-off than the default 0.5 raises recall.
- **Result on the untouched test part at 0.35:**

| Recall | Precision | Accuracy | Students invited |
|---|---|---|---|
| 0.81 (193 of 239 students with a PHQ-9 total of 10+) | 0.72 | 0.80 | 268 of 599 (45%) |

- **No neural network, no vector database.** With 2,992 rows and 26 inputs, logistic regression performs as well as the alternatives and is easier to explain. There was no document collection that needed searching.

### Results by group

A chi-square test shows a clear gender difference in PHQ-9 totals of 10 or more (33.8% for gender code 1, 45.4% for code 2, p < 0.001). The model's recall differs too:

- **Gender (survey codes):** recall about 0.76 for code 1 and 0.88 for code 2, a gap of about 12 points.
- **SES:** recall is highest for code 1 and lowest for code 6 (about 15 points apart). The middle codes do not form a smooth trend.
- **Ethnicity:** most groups are too small to judge, which is itself a limitation.

The model never sees gender or SES, but its inputs may differ between groups, so one cut-off flags fewer students from some groups. Group-specific cut-offs were considered and **not** used, because they would treat students differently by background. The design safeguard is that the PHQ-9 and the stage 2 rules do not depend on the model: if capacity allows, offering the PHQ-9 to every student removes this gap.

## Explainer (Gemini, Protocol C only)

- Gemini (`gemini-3.5-flash-lite`) through LangChain turns the stored list of high-scoring items into one or two neutral sentences. It adds nothing new.
- **Prompt engineering:** a system prompt with six rules (use only the items given; no diagnosis or cause; no prediction or treatment; no mention of suicide or self-harm; never the word "depression"; neutral tone) and two few-shot examples.
- **Guards in code:** crisis protocols never call the model; a list containing item 9 never calls it; an empty list returns a fixed sentence; any error returns a fixed fallback text; a 30-second timeout is set.
- **Testing:** five hand-written cases with a banned-word check, 13 real dataset rows, and failure and timeout tests. The author read every note. This is a small test, and the wording can vary slightly between runs.
- The note appears only when the counselor presses a button, and the list of items stays next to it so the note can be checked against its source.

## Tech stack

Python, pandas, SciPy, scikit-learn, Matplotlib and Seaborn (figures, in the notebook), LangChain with Google Gemini, Streamlit, Streamlit Community Cloud, Git and GitHub.

## Project structure

```
app.py                  Streamlit app (PHQ-9 tab and pre-screen tab)
main.py                 PHQ-9 rules, protocols, model training, explainer
stage3_new_dataset.ipynb  Data exploration, cleaning, statistics, model comparison
models/                 Saved pre-screen model
figures/                Charts used in the analysis
data/                   Codebook (and data files)
test_explainer.py       Tests for the explainer
test_gemini.py          First connection and prompt tests
requirements.txt        Packages needed to run the app
```

## Setup

1. Clone the repository and create a virtual environment.
2. Install the packages: `pip install -r requirements.txt`
3. Create a file named `.env` containing `GOOGLE_API_KEY=your-key` (never commit this file). Without a key the app still works, and the note shows a fixed fallback text.
4. Run the app: `streamlit run app.py`
5. To retrain the model: put the cleaned survey file at `data/DATASET-Raw_data_clean.csv` and run `python main.py`. The notebook also needs jupyter, matplotlib, seaborn and scipy.

## Ethics and limitations

- **This is not a diagnostic tool.** It supports a counselor, who decides. It must never be used to label a student, to decide alone about care, or to replace a conversation.
- **The pre-screen misses people.** At a cut-off of 0.35 about 19% of students with a PHQ-9 total of 10 or more were not invited in the test, and no cut-off reaches full recall unless everyone is invited. It cannot see item 9, so it cannot detect a crisis.
- **It recognizes a pattern, not depression.** The model learns correlated symptoms (stress and anxiety overlap with the PHQ-9). Its weights show how it uses a question, not causes.
- **Bias.** Leaving gender, SES and ethnicity out of the inputs does not guarantee fair results. Recall is lower for some groups (see above), and some groups are too small to assess.
- **Data limits.** One university, one voluntary survey in 2022, self-reported answers. A PHQ-9 total is a screening result, not a diagnosis. The survey may include minors.
- **Privacy.** The survey data is anonymous, so the protocols are demonstrated on data that cannot be linked to a person. A real deployment would need consent and an identified process. The app asks users not to enter identifying details, and the demo uses only invented example profiles.
- **Item 9 is imperfect.** It misses some people who do not report such thoughts, which is why the follow-up conversation, and not the score, is the safeguard.
- **The language model can phrase things wrongly.** That is why it is limited to the routine protocol, receives only item names, is guarded in code, and is checked by the counselor against the item list.
- **Public demo.** Anyone with the link can press the note button and use the free Gemini quota. If it runs out, the note falls back to fixed text and the rest keeps working.

## Author

Jenny Palti