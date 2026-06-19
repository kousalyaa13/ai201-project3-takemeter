# TakeMeter — Manifest Fandom Post Classifier

**AI201 · Project 3**

A text classifier that sorts posts from the *Manifest* TV-show fandom (Reddit) into how the viewer is **engaging** with the show. It compares a fine-tuned DistilBERT model against a zero-shot Groq baseline.

## Community & task

[*Manifest*](https://en.wikipedia.org/wiki/Manifest_(TV_series)) is an NBC/Netflix mystery drama about the passengers of Flight 828, who disappear and return 5½ years later with visions called "Callings." Because the show is one long mystery, fans engage in three clearly different ways, which become the labels:

| Label | Meaning |
|---|---|
| `theory` | A hypothesis or prediction about the show's mysteries. |
| `rant_rave` | A mainly emotional reaction — liking or disliking a character, relationship, or plot point. |
| `plot_question` | A factual question about something the viewer missed or found confusing. |

Full label definitions, edge cases, and the annotation rules are in [planning.md](planning.md).

## Dataset

- **Source:** Reddit (r/Manifest and related Manifest threads). Public posts only.
- **Size:** 210 examples — `plot_question` 72, `rant_rave` 70, `theory` 68.
- **File:** [manifest_reviews.csv](manifest_reviews.csv) (`text`, `label`, `source`, `notes`).
- **Split:** 70 / 15 / 15 train / val / test, stratified by label (handled by the notebook).
- The `notes` column flags borderline posts; the 19 hardest are also collected in [ambiguous_review.csv](ambiguous_review.csv).

## How to run

Open the notebook in Google Colab with a **T4 GPU** runtime, upload `manifest_reviews.csv`, add your `GROQ_API_KEY` to Colab Secrets, and run the cells top to bottom. Fine-tuning takes 5–15 minutes.

---

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Zero-shot baseline (Groq, `llama-3.3-70b-versatile`) | 0.875 | 0.87 |
| Fine-tuned DistilBERT | 0.844 | 0.85 |

On this test set the **fine-tuned model slightly underperformed the zero-shot baseline** (0.85 vs 0.87 macro-F1). See the comparison below — the result is more nuanced than the headline number.

### Baseline (zero-shot Groq) — reflection

**Overall:** 0.875 accuracy, 0.87 macro-F1 on the 32-example test set. All 32 responses parsed cleanly (no unparseable outputs).

Per-class metrics:

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| theory | 0.91 | 1.00 | 0.95 | 10 |
| rant_rave | 0.79 | 1.00 | 0.88 | 11 |
| plot_question | 1.00 | 0.64 | 0.78 | 11 |

**Where it struggled:** the weakness is entirely in **`plot_question` recall (0.64)** — it caught only 7 of 11 factual questions. The other two labels had perfect recall. Since `plot_question` precision was 1.00, the problem is one-directional: real questions are **leaking out** into the other labels, not the reverse.

**Where they went:** `rant_rave` precision dropped to 0.79 (~3 false positives) and `theory` to 0.91 (~1 false positive) — accounting for exactly the 4 missed questions:
- ~3 questions misread as **`rant_rave`** → likely *complaints shaped like a question* ("Why is the casting so terrible?").
- ~1 question misread as **`theory`** → a *question that's really arguing a point*.

This is exactly the boundary [planning.md](planning.md) predicted: the baseline leans on emotional tone and lore content instead of the question's actual intent.

**Hypothesis to test after fine-tuning:** fine-tuning should **raise `plot_question` recall** and tighten `rant_rave`/`theory` precision, because the model will learn the surface patterns of genuine questions rather than guessing from tone. If `plot_question` recall stays low, that points to too few question examples on the hard boundary — a data problem, not a model problem.

### Fine-tuned DistilBERT — results

**Overall:** 0.844 accuracy, 0.85 macro-F1 on the 32-example test set.

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| theory | 1.00 | 0.80 | 0.89 | 10 |
| rant_rave | 0.69 | 1.00 | 0.81 | 11 |
| plot_question | 1.00 | 0.73 | 0.84 | 11 |

**Confusion matrix** (rows = true label, columns = predicted):

| true ↓ / pred → | theory | rant_rave | plot_question | total |
|---|---|---|---|---|
| **theory** | 8 | 2 | 0 | 10 |
| **rant_rave** | 0 | 11 | 0 | 11 |
| **plot_question** | 0 | 3 | 8 | 11 |
| **total predicted** | 8 | 16 | 8 | 32 |

The matrix makes the pattern unmistakable: **`rant_rave` is the only column with off-diagonal entries.** Every one of the 5 errors is `theory → rant_rave` (2) or `plot_question → rant_rave` (3). Nothing is ever misclassified *as* `theory` or *as* `plot_question` (both columns are clean → precision 1.00). The model never confuses `theory` with `plot_question` directly; instead both bleed in one direction into `rant_rave`.

![Confusion matrix](confusion_matrix.png)

**What the model learned:** all 5 errors were predicted **`rant_rave`** (2 true `theory`, 3 true `plot_question`), and every one came at very low confidence (0.34–0.38, barely above the 0.33 random floor for 3 classes). `rant_rave` precision fell to 0.69 while its recall hit 1.00 — i.e. the model uses `rant_rave` as a **fallback bucket** for posts it can't confidently place. `theory` and `plot_question` both kept perfect precision (1.00): when the model *does* commit to those labels it is right, it just commits too rarely.

### Comparison

| Metric | Baseline | Fine-tuned | Change |
|---|---|---|---|
| Macro-F1 | 0.87 | 0.85 | ▼ 0.02 |
| `plot_question` recall | 0.64 | 0.73 | ▲ 0.09 |
| `rant_rave` precision | 0.79 | 0.69 | ▼ 0.10 |
| `theory` recall | 1.00 | 0.80 | ▼ 0.20 |

**Was the hypothesis confirmed? Partially.** I predicted fine-tuning would raise `plot_question` recall — and it did (0.64 → 0.73). But it did **not** improve overall: the model traded that gain for a new, worse failure mode. Where the *baseline* leaked questions into multiple labels based on tone, the *fine-tuned* model collapses almost everything it's unsure about into `rant_rave`, now also pulling in clean `theory` posts (`theory` recall dropped 1.00 → 0.80).

**Why this happened — the data problem the hypothesis flagged.** With only ~150 training examples, DistilBERT didn't learn the `theory`/`plot_question` boundary well enough to be confident, so it defaults to the largest, most lexically varied class (`rant_rave`). The near-random confidences (~0.34) confirm under-training, not genuine ambiguity. This matches the planning.md prediction: if `plot_question` recall stayed weak after fine-tuning, the bottleneck is **too few examples on the hard boundary**, not the model. The fix is more data (especially emotionally-worded questions and plainly-stated theories), not more epochs.

---

## Error analysis

All 5 of the fine-tuned model's mistakes were predicted `rant_rave` at low confidence. I analyzed three that show the *different reasons* this happened.

**1. Emotional framing on a factual question** — `plot_question` → `rant_rave` (conf 0.34)
> "I'm still mad I never found out what happened to the comatose passengers. Did they wake up? How did the government explain their disappearances? Where are they now? I have so many questions."

This is a cluster of genuine factual questions, but it opens with "I'm still mad." The model latched onto the emotional lead word and bucketed it as a rant. This is exactly the `rant_rave` vs `plot_question` edge case from planning.md — and I had flagged this specific post in [ambiguous_review.csv](ambiguous_review.csv). The lesson: emotional *tone* is overriding the question *intent*, which means I need more training examples of frustrated-but-factual questions.

**2. A question that wraps a theory** — `plot_question` → `rant_rave` (conf 0.34)
> "How did the major know that the passengers' minds were connected as soon as they landed? Remember: in season 1 she kidnapped the passengers... This suggests that the government was involved..."

Per my annotation rule I labeled this `plot_question` (the lead intent is "how did she know?"), but it also argues a point, so it sits right on the `theory`↔`plot_question` line. The model didn't pick `theory` *or* `plot_question` — it gave up and chose `rant_rave` at near-random confidence. This is the boundary planning.md predicted would be hardest, and it confirms the model lacks enough examples to resolve it.

**3. A clean theory the model still missed** — `theory` → `rant_rave` (conf 0.35)
> "My theory is the sapphire is just the physical anchor for the Callings. Whoever holds it can broadcast or block them, which is exactly why Angelina could fake them once she got a shard."

This is the *surprising* one: it literally starts with "My theory is" and lays out a clear mechanism — it is not a genuine edge case. The model misclassifying it (and at 0.35 confidence) shows the failure isn't really about ambiguity; it's **under-training**. With only ~150 examples the model never built a confident `theory` representation, so a textbook theory still fell into the `rant_rave` fallback bucket. This is the strongest evidence that the fix is more data, not better labels.

**Labeling problem or data problem?** I re-read all 5 misclassified posts against my planning.md definitions and confirmed my original labels were correct and *consistent* (e.g. I labeled every "I'm still mad… [questions]" post as `plot_question`, not just this one). So this is **not annotation inconsistency** — it's a training-data problem: too few examples teaching the model that emotional wording and embedded reasoning don't change a post's primary intent.

**What would fix it:** more training examples on exactly the confused boundary — specifically (a) factual questions written with strong emotion ("I'm so mad, why did…"), and (b) plainly-stated theories ("My theory is…"), so the model stops treating any emotional or argumentative wording as a `rant_rave` signal. A tighter label definition would *not* help here, because my definitions already separate these cleanly; the model simply hasn't seen enough of the hard cases. More epochs would not help either — the near-random confidences point to too little signal, not too little training time.

---

## Sample classifications

A few test posts run through the **fine-tuned** model, with the predicted label and the model's confidence (softmax probability of the predicted class):

| Post (truncated) | Predicted | Confidence | Correct? |
|---|---|---|---|
| "I wanted to like this show so badly but after watching the first season and then into the second I just couldn't do it anymore. So many actors got on my nerves." | rant_rave | 0.40 | ✅ (true: rant_rave) |
| "I stopped watching because of the terrible acting at times, plus when Netflix took over, the plot line got even worse and nonsensical." | rant_rave | 0.39 | ✅ (true: rant_rave) |
| "I'm still mad I never found out what happened to the comatose passengers. Did they wake up?…" | rant_rave | 0.34 | ❌ (true: plot_question) |
| "My theory is the sapphire is just the physical anchor for the Callings…" | rant_rave | 0.35 | ❌ (true: theory) |
| "How did the major know that the passengers' minds were connected as soon as they landed?…" | rant_rave | 0.34 | ❌ (true: plot_question) |

**Why a correct prediction is reasonable:** the model labeled "I wanted to like this show so badly… So many actors got on my nerves." as `rant_rave` — the right call, because the post is pure emotional reaction (disappointment, frustration with the acting) and makes no factual question or hypothesis, which is exactly the `rant_rave` definition. The post also looks like the kind of opinionated, first-person text that dominates the `rant_rave` training examples.

> **The bigger signal — there is barely any confidence gap.** The model's *correct* predictions top out at ~0.40, only just above its *errors* at ~0.34, and both sit near the 0.33 random floor for three classes. So the model isn't confidently right and uncertainly wrong — it's **under-confident across the board**, which is strong evidence of under-training on a small dataset. Notably, all of its highest-confidence correct predictions are `rant_rave`, confirming that `rant_rave` is the model's default lean (both its surest correct calls *and* its error sink point to the same class).

## Reflection: what the model captured vs. what I intended

**What I intended** each label to capture (from planning.md) was the *primary intent* of a post — is the author theorizing, reacting, or asking? — regardless of surface features like punctuation or emotional wording.

**What the model actually learned** is narrower and more surface-level:
- It captured the **prototypical center of `rant_rave`** best: clearly opinionated, first-person posts ("I wanted to like this show so badly…") are classified correctly, and they are the only posts the model is even mildly confident about (~0.40). It did *not* build an equally strong center for `theory` or `plot_question` — it even missed a textbook theory ("My theory is the sapphire…"), so its grasp of those classes is correct only on the easiest cases and never confident.
- It **overfit to emotional and argumentative wording as a `rant_rave` cue.** Because `rant_rave` is the largest and most lexically varied class, the model effectively learned "if a post sounds emotional or opinionated, it's a rant" — so a factual question that happens to start with "I'm so mad" or a theory phrased with conviction gets pulled into `rant_rave`.
- It **missed the intent-over-form rule entirely**, which is the exact thing my label definitions are built on. The model has no representation of "this is a question *despite* the emotion" or "this is an argument *despite* the question mark." That gap is the whole `theory`/`plot_question` → `rant_rave` leak.

In short: my labels are defined by *intent*; the model learned *tone and vocabulary*. The decision boundary it actually drew is "emotional/opinionated vs. neutral," which is close to but not the same as the three-way intent split I designed — and the difference is exactly where it fails.

## Spec reflection

**One way the spec helped:** the spec required me to write out **hard edge cases and an explicit annotation rule before collecting data**. Forcing the "label by main purpose, not punctuation" rule up front did two things: it kept my 210 labels consistent (so I could later rule out annotation error as the cause of mistakes), and it let me *predict the exact failure boundary* — I wrote in planning.md that `theory`↔`plot_question` would be hardest, and that's precisely where both models broke. The structured planning step turned the error analysis from guesswork into hypothesis-testing.

**One way my implementation diverged:** planning.md originally specified a `pre_labeled` column to track LLM-assisted pre-labeling for disclosure. In implementation I **dropped pre-labeling entirely** and labeled every post from scratch (the CSV has a `notes` column instead). Why: the hardest, most informative posts are the boundary cases, and letting an LLM pre-label them risked anchoring my own judgment on exactly the examples where independent judgment matters most. Labeling unaided kept the ground truth clean, at the cost of more manual effort.

## Hyperparameters

Used the notebook defaults (3 epochs, learning rate 2e-5, batch size 16). 

---

## AI usage

I labeled every example in the dataset myself and did **not** use AI to pre-label any post (see the spec-reflection divergence above). Specific instances of AI assistance:

**1. Dataset cleanup and CSV normalization.**
- *What I directed:* take my Reddit posts (grouped by label with `--` section headers) and align every row to `text,label,source,notes`, apply each section's label, fix CSV quoting, and flag any post that straddled two labels.
- *What it produced:* a script that normalized 210 rows, fixed two real bugs (a stray trailing quote and an unescaped `"holy grail"` that broke CSV parsing), and added `notes` flags on 19 borderline posts.
- *What I changed/overrode:* I reviewed all 19 flags and overrode the AI's placement on two — I moved a real-world-production question from `theory` → `plot_question`, and an emotional musing from `plot_question` → `rant_rave`. I left the other 17 as I had originally labeled them.

**2. Groq baseline prompt.**
- *What I directed:* write the Section 5 classification prompt using my planning.md label definitions, one example per label, with a strict "output only the label name" instruction.
- *What it produced:* the `SYSTEM_PROMPT`, including an extra "judge by main purpose" hard-cases block derived from my edge cases.
- *What I changed/overrode:* I verified the three example posts were real ones from my dataset and confirmed the output instruction matched the notebook's parser (lowercase, exact label match) — it parsed 32/32 responses with zero failures.

**3. Failure-pattern analysis (post-evaluation).**
- *What I directed:* look across the fine-tuned model's 5 wrong predictions and propose error patterns.
- *What it produced:* the observation that all 5 errors collapse into `rant_rave` at near-random confidence, and the data-vs-labeling diagnosis.
- *What I changed/overrode:* I verified the pattern myself by re-reading each misclassified post against my definitions and checking I had labeled similar posts consistently, before accepting it into this report.

I also used an AI assistant to draft and tighten the prose in this README and planning.md, which I reviewed and edited.
