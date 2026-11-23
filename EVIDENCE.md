# Evidence sheet: the numbers that show the system is credible and safe

Student Support Prioritizer. One page to scan. Each number is followed by what it actually proves — and the last section is what it does not.

## 1. Does the ordering work?

Measured on the test part: **599 students, 166 of whom answered above 0 on PHQ-9 item 9.**

| Number | What it proves |
|---|---|
| Recall **0.73**, precision **0.53**, accuracy **0.74** at the 0.30 boundary | The ordering finds most of the students who reported thoughts on item 9, and about half the students it puts early turn out to have reported them |
| Waves 1 + 2 = **229 students (38% of the cohort)** holding **121 of 166 (73%)** | The first 38% of the queue contains nearly three quarters of the affected students |
| Working alphabetically, 38% of the cohort reaches about **38%** of them | The ordering roughly doubles how fast a counselor reaches them |
| Within-wave rates **0.63 / 0.42 / 0.12** | The three waves are genuinely different populations, not an arbitrary slicing |

## 2. Is it reliable, or could it be a fluke?

| Number | What it proves |
|---|---|
| **80/20 stratified split**, `random_state=42`, 2,393 training / 599 test | Standard, reproducible, and the class balance is preserved in both parts |
| Boundaries chosen with **5-fold `cross_val_predict` on the training part only** (2,393 students, 661 positive) | Every training student got a probability from a model that never saw them |
| The test part was looked at **once**, after the boundaries were already fixed | No tuning on test data |
| Cross-validation predicted **65 / 40 / 13**; the untouched test gave **63 / 42 / 12** | The headline result. The wave structure means the same thing on students the model had never seen |

## 3. Is it safe?

These are design facts rather than measurements, and they are what make the performance numbers acceptable rather than alarming.

| Fact | What it protects against |
|---|---|
| Every student is offered the PHQ-9 regardless of wave | The model can never prevent anyone from being assessed |
| Item 9 is read **directly from the student's own answer** by a fixed rule | The most important signal in the tool never depends on a prediction |
| Any answer above 0 on item 9 triggers an **escalation protocol, which governs the response**; the PHQ-9 total is shown beside it as context, not as a substitute | The protocol decides what happens. The counselor is told the decision is theirs and is not asked to make it with half the data hidden |
| The language model **never receives or mentions PHQ-9 item 9** — it is stripped from the list sent to Gemini, and the item 9 sentence is fixed text added by code afterwards | The most sensitive fact in the tool cannot be paraphrased, softened or invented by a model |
| **No probability or percentage is ever shown in the app** | A counselor cannot read an inferred number as a fact about a particular student |
| The further-reading section displays only **passages from a fixed, author-vetted corpus**; the model selects which to show and writes none of them | What a counselor reads as guidance is always published text that was reviewed before it shipped |
| A **similarity floor of 0.68** suppresses any passage below it, so a level can legitimately show nothing | Retrieval can stay silent, but it cannot present an irrelevant passage as if it were guidance |

## 4. Do you know where it is unfair?

| Number | What it proves |
|---|---|
| Gender recall **0.62 against 0.81** — a 19-point gap | Measured rather than assumed from leaving the column out |
| Early-wave placement **28% against 46%** | The size of the effect is stated, not just its existence |
| SES recall **0.90 (code 1) down to 0.42 (code 6)** — but code 6 holds **only 12 affected students**, so 0.42 is 5 of 12 | Sample sizes are reported and no conclusion is drawn from small groups |
| Ethnicity category 1 holds **3 affected students** | The audit's own limits are stated rather than hidden |
| **No group-specific boundaries** | A deliberate choice not to treat students differently by background |

Because every student is offered the PHQ-9, the gap changes **when** a student is assessed, not **whether**.

## 5. How the similarity floor was chosen

| Number | What it proves |
|---|---|
| Every protocol step was scored against all **20 corpus passages** and the results printed before any threshold was picked | The number came from measurement, not from a plausible-looking default |
| Lowest genuine match **0.684**; highest false match **0.677** — one step ("move to Protocol A at once") describes the tool's own internal routing and has no passage behind it | The floor sits between the two, which is the only defensible place for it |
| At 15 passages the ordering was inverted — the false match (0.677) outscored two genuine ones — and only separated cleanly after the corpus grew to 20 | The condition under which the method fails is stated, rather than claiming it always works |

## 6. What these numbers do not prove

- **The model misses about one in four.** At 0.30 it misses 45 of 166 in the test part, and no boundary fixes it — even at 0.15 it still missed 76 in training. This is the reason it orders and never gates.
- **Precision is 0.53.** Roughly half the students placed in the early waves did not report thoughts on item 9. In a queue that costs waiting time, not care — which is only acceptable because it is a queue.
- **One university, one voluntary survey, 2022, self-reported.** Nothing here shows it would transfer to another institution or another year.
- **The explainer is tested, not evaluated.** About 18 cases, a rough banned-word check, and every note read by hand. It runs on all six levels, with item 9 excluded from what it sees, but there is no scored evaluation behind it.
- **The retrieval corpus is 20 passages.** Small enough that a flat scan is the right implementation and a vector database would be theatre, but also small enough that coverage is uneven: levels 1 to 4 draw on crisis-response material, levels 5 and 6 on a handful of non-crisis passages. The floor's margin is 0.007, which is thin.
- **The retrieval is not evaluated either.** The top matches were read and judged sensible by the author; there is no scored relevance measure.
- **The PSS-10 bands are convention**, not validated clinical cut-points like the GAD-7's. Nothing in the tool depends on them.

## If you only remember three things

1. **63 / 42 / 12 against a predicted 65 / 40 / 13** — it replicated on data the model had never seen.
2. **38% of the queue holds 73% of them** — roughly double what any arbitrary order would reach.
3. **Recall 0.73 is exactly why it only sets the order** — the model's weakness is the reason for the design, not an embarrassment to be explained away.