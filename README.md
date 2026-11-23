# Student Support Prioritizer

A prioritization aid that helps counselors decide which students to see first, based on screening answers. **It does not diagnose, it does not decide who gets assessed, and the counselor decides.**

**Live app:** https://student-support-prioritizer.streamlit.app/

Capstone project for the GenAI & Machine Learning Bootcamp (Developers Institute).

## What it does

Every student is assessed. The tool only sets the order.

The tool works in two stages.

**Stage 1: priority ordering.** A logistic regression model reads 17 answers the student has already given on two standard questionnaires — the PSS-10 stress scale and the GAD-7 anxiety scale — and places the student in one of three waves. The waves tell a counselor who to work through first. The app shows the wave and what to do in it. It never shows a probability, a percentage or a risk score.

**Stage 2: PHQ-9 follow-up.** Fixed rules turn the nine PHQ-9 answers into one of six follow-up levels and one of three protocols. Each result lists which items scored high, so the counselor can see why a student was flagged. For the support protocol only, a Gemini model can write a short plain-language note restating those items.

### The three waves

| Wave | Screening | Session |
|---|---|---|
| First | Immediately administer PHQ-9. | Immediately invite for a session and administer C-SSRS. |
| Second | Immediately administer PHQ-9. | Invite for a session and administer C-SSRS after the first wave, guided by the results of those assessments. |
| Third | PHQ-9 offered as part of standard screening. | A session if the PHQ-9 indicates one, or if the student asks. |

The C-SSRS sits in the session and not in the screening, because it is administered by an interviewer.

**A queue, not a gate.** Every student is offered the PHQ-9 whatever their wave. Item 9 — the question about thoughts of being better off dead or of self-harm — is therefore always asked directly, and the crisis rule always fires on the student's own answer, never on a prediction. The model changes *when* a student is seen. It never changes *whether*.

### Follow-up levels (stage 2)

| Level | Label | PHQ-9 answers | Protocol |
|---|---|---|---|
| 1 | Same day | item 9 = 3 | A: Immediate safety follow-up |
| 2 | Within 24-48 hours | item 9 = 2 | A: Immediate safety follow-up |
| 3 | Priority review | item 9 = 1 and total 15 or higher | B: Clinician review |
| 4 | Scheduled review | item 9 = 1 and total 14 or lower | B: Clinician review |
| 5 | Check-in offered | item 9 = 0 and total 15 or higher | C: Support and check-in |
| 6 | Standard | item 9 = 0 and total 14 or lower | C: Support and check-in |

The split between levels 3 and 4 is a sorting aid, not a difference in risk.

## Safety design

- **Crisis rule.** If item 9 is above 0, the app shows the escalation protocol and **no score**.
- **The most important signal is a rule, not a model.** Item 9 is read directly from the student's own answer. A model can be wrong in the direction that costs a life, so nothing about the crisis path depends on a prediction.
- **No language model near a crisis.** Protocols A and B use fixed text only. The Gemini note exists only for Protocol C, and the code blocks it even there if item 9 appears in the list.
- **The ordering model cannot see item 9**, so it cannot detect a crisis. A third-wave placement never means "no risk", and the app says so every time.
- The protocol texts were adapted from school suicide-intervention procedures (Mental Health Commission of Canada national guide, NYSED, Pasadena USD, Sherwood SD) and reviewed and finalized by the author, who has a clinical psychology background. The timeframes are the author's own choices.
- The counselor decides. The app asks users not to enter names or identifying details.

## Data

- Source: the 2022 wellbeing survey of the Icesi University community (Colombia). The survey states that its data is anonymous and used for academic purposes. The codebook is in `data/`.
- The cleaned file has 2,992 students. The project uses survey questions 16, 17 and 18 only: PSS-10 stress items (0-4), GAD-7 anxiety items (0-3) and the PHQ-9 (0-3).
- **The survey data file itself is not in this repository.** Only the codebook is included. The cleaned file has to be supplied locally to retrain the model (see Setup).
- Of the 2,992 students, **827 (27.6%) answered above 0 on PHQ-9 item 9.**
- The demographic questions — gender, socioeconomic code and ethnicity — are loaded but kept out of the model's inputs. They are used only afterwards, to check results by group.
- The earlier sections of the survey (demographics and the "yesterday" wellbeing questions) were dropped from the project entirely. They are not a scored instrument: the survey defines no totals or cut-offs for them.
- Cleaning: 11 impossible birth years and about 120 parent-education values above 25 were noted. No rows were removed. None of those columns are used.
- The project started with a different dataset (Student Depression Dataset) and switched to this survey after instructor feedback, because the survey contains the full PHQ-9 alongside validated questionnaires.

## Model (stage 1)

- **Inputs (17):** `pss1`-`pss10` and `gad1`-`gad7`. Nothing else.
- **Target:** whether PHQ-9 item 9 was answered 1, 2 or 3 — that is, above 0.
- **The PHQ-9 items are never inputs.** This is the circularity test: a model must not need something that would not exist at the moment you want to use it. If the PHQ-9 were an input, the model would be predicting an answer from itself, and it would only work once the questionnaire it is meant to prioritize had already been completed.
- **Demographics are excluded from learning and kept for auditing.** That separation is deliberate.
- **Split:** 80% training (2,393 students, 661 of them answered above 0 on item 9), 20% test (599), stratified, `random_state=42`.
- **Model:** logistic regression — readable weights, no overfitting, and fast enough to retrain in seconds. An earlier stage of the project compared it against a random forest on the previous target; the two were tied in cross-validation and the random forest overfitted (training accuracy 1.00), so logistic regression was chosen and that choice was kept.
- **Wave boundaries:** probability 0.50 and 0.30. Above 0.50 is the first wave, 0.30 to 0.50 the second, below 0.30 the third.
- **No neural network, no vector database.** With 2,992 rows and 17 inputs, logistic regression performs as well as the alternatives and is far easier to explain. There was no document collection that needed searching.

### Why 17 questions to decide whether to ask 9

The ordering is not more efficient in questions asked, and is not claimed to be. It is built for institutions that already run the PSS-10 and the GAD-7 in a wellbeing survey, where those answers exist at no extra cost. The core of the tool is the PHQ-9 with its levels and crisis rule, which works on its own without any model at all.

## How the wave boundaries were chosen

`choose_cutoff.py` runs `cross_val_predict` with 5 folds on the **training part only**, so every training student receives a probability from a model that never saw them. Cut-offs were swept from 0.50 down to 0.15.

Training sweep (2,393 students, 661 positive):

| Cut-off | Invited | Rate | Caught | Missed | Recall | Precision |
|---|---|---|---|---|---|---|
| 0.50 | 442 | 0.18 | 287 | 374 | 0.43 | 0.65 |
| 0.45 | 542 | 0.23 | 332 | 329 | 0.50 | 0.61 |
| 0.40 | 635 | 0.27 | 371 | 290 | 0.56 | 0.58 |
| 0.35 | 746 | 0.31 | 418 | 243 | 0.63 | 0.56 |
| 0.30 | 883 | 0.37 | 464 | 197 | 0.70 | 0.53 |
| 0.25 | 1026 | 0.43 | 513 | 148 | 0.78 | 0.50 |
| 0.20 | 1193 | 0.50 | 550 | 111 | 0.83 | 0.46 |
| 0.15 | 1403 | 0.59 | 585 | 76 | 0.89 | 0.42 |

Down to 0.25, each extra student reached early costs about 3 extra places in the early waves. Below that the cost rises to 4.5 and then 6.0. That bend is the reason for not going lower.

**The test part was looked at once, afterwards, with the boundaries already fixed.** Moving them now would mean tuning on test data, so they stay where they are.

## Results

Measured on the untouched test part: 599 students, 166 of whom answered above 0 on item 9.

At the 0.30 boundary: **recall 0.73, precision 0.53, accuracy 0.74.**

| Wave | Students | Share | Answered above 0 | Rate within wave |
|---|---|---|---|---|
| 1 | 118 | 20% | 74 | 0.63 |
| 2 | 111 | 19% | 47 | 0.42 |
| 3 | 370 | 62% | 45 | 0.12 |

**Waves 1 and 2 together are 229 students — 38% of the cohort — and they contain 121 of the 166 affected students, or 73%.** Working alphabetically instead, 38% of the cohort would reach about 38% of them. The ordering roughly doubles how quickly a counselor reaches the students who answered above 0.

**The structure replicated.** Cross-validation on the training part predicted within-wave rates of 65 / 40 / 13 percent. The untouched test part gave 63 / 42 / 12. The waves mean on new students what they meant in training, which is the result this project rests on.

### What separates the waves

The GAD-7 total separates the first wave from the second (a typical first-wave student scores 14 of 21, a typical second-wave student 7). The PSS-10 total separates the second wave from the third (26 of 40 against 16). This matches the model's weights, where `gad5`, `gad3` and `gad4` are among the strongest positive ones.

A typical third-wave student still scores 16 on the PSS-10, which is inside the moderate range. The waves are not simply "stressed students against unstressed ones".

### Results by group

Before looking at the model, it is worth noting that the rates differ by gender in the data itself. A chi-square test across all 2,992 students, on PHQ-9 totals of 10 or more, gives 33.8% for gender code 1 (476 of 1,407) against 45.4% for code 2 (719 of 1,585), p < 0.001. That is a difference in the survey answers, not something the model introduced — and it says nothing about why, whether symptoms, response style, or who chose to answer the survey at all.

The model's own results, checked on the test part using the demographic columns it never saw:

| Gender | Students | Answered above 0 | Placed in waves 1-2 | Recall |
|---|---|---|---|---|
| 1 | 265 | 68 | 28% | 0.62 |
| 2 | 334 | 98 | 46% | 0.81 |

That is a 19-point recall gap, and gender 2 students enter the early waves at nearly twice the rate.

The model places gender 1 students in the early waves less often. Because every student is offered the PHQ-9 regardless of wave, this gap affects *when* a student is assessed, not *whether* they are asked. A student the model ranks low is still asked item 9 directly, and the crisis rule applies to their answer.

**Group-specific boundaries were considered and not used**, because they would treat students differently according to their background.

Socioeconomic codes: recall runs from 0.90 (code 1) down to 0.42 (code 6). Code 6 contains only 12 affected students, so 0.42 means 5 of 12 — too few to conclude anything from. Ethnicity category 1 contains 3 affected students. These numbers are reported with their sample sizes attached and no conclusions are drawn from the small groups. That the audit cannot assess small groups is itself a limitation.

## Scoring, and where every number comes from

The Icesi survey defines no scoring system of its own; its codebook gives response codes per item. So every number in this tool comes from one of three places — published instrument scoring, the author's own rules, or nothing at all.

**From the published instruments:**

- PHQ-9 bands (total 0-27): 0-4 Minimal, 5-9 Mild, 10-14 Moderate, 15-19 Moderately severe, 20-27 Severe, and the total of 10 or more as a screening threshold — Kroenke, Spitzer & Williams (2001). In this dataset the bands hold 861 / 936 / 609 / 358 / 228 students.
- GAD-7 cut points 5, 10 and 15 — Spitzer et al. (2006).
- PSS-10 reverse scoring of items 4, 5, 7 and 8 — Cohen et al. (1983).

**The author's own rules, and stated as hers:** the six follow-up levels and their ordering; the three protocols and their step text; the two-tier crisis rule; the convention that an item scoring 2 or more counts as a reason; and the wave boundaries.

**One caveat worth stating plainly.** The PSS-10 bands used here (0-13 low, 14-26 moderate, 27-40 high) are a widely repeated convention from templates and trial documents, not validated clinical cut-points in the way the GAD-7's are. They are used for description only, and nothing in the tool depends on them.

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
app.py                    Streamlit app (PHQ-9 tab and pre-screen tab)
main.py                   PHQ-9 rules, protocols, model training, explainer
choose_cutoff.py          Cross-validated sweep used to set the wave boundaries
stage3_new_dataset.ipynb  Data exploration, cleaning, statistics, model comparison
models/                   Saved pre-screen model
figures/                  Charts used in the analysis
data/                     Codebook
test_explainer.py         Tests for the explainer
test_gemini.py            First connection and prompt tests
requirements.txt          Packages needed to run the app
```

## Setup

1. Clone the repository and create a virtual environment.
2. Install the packages: `pip install -r requirements.txt`
3. Create a file named `.env` containing `GOOGLE_API_KEY=your-key` (never commit this file). Without a key the app still works, and the note shows a fixed fallback text.
4. Run the app: `streamlit run app.py`
5. To retrain the model: put the cleaned survey file at `data/DATASET-Raw_data_clean.csv` and run `python main.py`. To reproduce the boundary sweep, run `python choose_cutoff.py`. The notebook also needs jupyter, matplotlib, seaborn and scipy.

Note that the app caches the loaded model in memory. After retraining and redeploying, restart the app so it picks up the new file.

## Ethics and limitations

- **This is not a diagnostic tool.** It supports a counselor, who decides. It must never be used to label a student, to decide alone about care, or to replace a conversation.
- **The ordering is imperfect, which is why it is only an ordering.** At the 0.30 boundary the model misses about one in four of the students who answered above 0 on item 9, and no boundary fixes that — even at 0.15 it still misses 76 of them in training. This is precisely why the tool sets the order and not the entry: a student placed in the third wave is still offered the PHQ-9 and still asked item 9 directly.
- **It recognizes a pattern, not depression.** Stress and anxiety overlap with the PHQ-9 in content, and the model learns that overlap. Its weights show how it uses a question, not what causes anything.
- **Bias.** Leaving gender, socioeconomic code and ethnicity out of the inputs does not guarantee equal results. Recall is lower for gender code 1 (0.62 against 0.81), and some groups are too small to assess at all.
- **No probability is ever shown in the app.** The within-wave rates reported above are documentation, not something a counselor sees. Telling a counselor that a given share of a wave answered above 0 would disclose inferred sensitive information about an individual student.
- **Data limits.** One university, one voluntary survey in 2022, self-reported answers. A PHQ-9 total is a screening result, not a diagnosis. The survey may include minors.
- **Privacy.** The survey data is anonymous, so the protocols are demonstrated on data that cannot be linked to a person. A real deployment would need consent and an identified process. The app asks users not to enter identifying details, and the demo uses only typical answer patterns rather than real students.
- **Item 9 is imperfect.** It misses some people who do not report such thoughts, which is why the follow-up conversation, and not the score, is the safeguard.
- **The language model can phrase things wrongly.** That is why it is limited to the support protocol, receives only item names, is guarded in code, and is checked by the counselor against the item list.
- **Public demo.** Anyone with the link can press the note button and use the free Gemini quota. If it runs out, the note falls back to fixed text and the rest keeps working.
- **A real deployment** must insert the institution's own emergency and crisis numbers in place of the placeholder wording in the protocols.

## Author

Jenny Palti