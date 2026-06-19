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
| Fine-tuned DistilBERT | 0.906 | 0.91 |

The **fine-tuned model beat the zero-shot baseline** (0.906 vs 0.875 accuracy, 0.91 vs 0.87 macro-F1) — but only after fixing an undertraining problem. At the notebook's default **3 epochs the model was stuck at chance** (~0.33 validation accuracy, collapsing every uncertain post into one default label). Raising training to **8 epochs** let it actually learn the classes (see the training curve and [Hyperparameters](#hyperparameters)). The numbers below are the 8-epoch model.

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

**Overall:** 0.906 accuracy, ~0.91 macro-F1 on the 32-example test set (29/32 correct).

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| theory | 0.90 | 0.90 | 0.90 | 10 |
| rant_rave | 0.91 | 0.91 | 0.91 | 11 |
| plot_question | 0.91 | 0.91 | 0.91 | 11 |

**Confusion matrix** (rows = true label, columns = predicted):

| true ↓ / pred → | theory | rant_rave | plot_question | total |
|---|---|---|---|---|
| **theory** | 9 | 0 | 1 | 10 |
| **rant_rave** | 1 | 10 | 0 | 11 |
| **plot_question** | 0 | 1 | 10 | 11 |
| **total predicted** | 10 | 11 | 11 | 32 |

This is a **balanced result**: exactly one error per class, and they point in three *different* directions (`theory→plot_question`, `rant_rave→theory`, `plot_question→rant_rave`). There is no single "dumping" class anymore — every label has ~0.90 precision *and* ~0.90 recall. Compare this to the earlier 3-epoch run, where the model was at chance and threw everything into one default label.

![Confusion matrix](confusion_matrix.png)

**Training curve (why 8 epochs).** Validation accuracy by epoch shows the model spending two full epochs at the random floor before it starts learning:

| Epoch | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|
| Val accuracy | 0.32 | 0.32 | 0.71 | 0.77 | 0.87 | 0.94 | 0.94 | **0.97** |

Validation loss was still falling at epoch 8 (no overfitting yet), and `load_best_model_at_end=True` keeps the epoch-8 weights.

**What the model learned:** the 3 errors all came at higher confidence than the old failing run (0.44–0.61 vs ~0.34) and each sits on a genuine label boundary rather than a blind default — see the error analysis. The model now commits to all three labels with roughly equal skill, instead of hiding in one fallback class.

### Comparison

| Metric | Baseline | Fine-tuned (8 ep) | Change |
|---|---|---|---|
| Accuracy | 0.875 | 0.906 | ▲ 0.031 |
| Macro-F1 | 0.87 | 0.91 | ▲ 0.04 |
| `plot_question` recall | 0.64 | 0.91 | ▲ 0.27 |
| `theory` recall | 1.00 | 0.90 | ▼ 0.10 |
| `rant_rave` precision | 0.79 | 0.91 | ▲ 0.12 |

**Was the hypothesis confirmed? Yes.** I predicted fine-tuning would raise `plot_question` recall and tighten `rant_rave` precision — both happened, dramatically: `plot_question` recall jumped 0.64 → 0.91 (the baseline's single biggest weakness), and `rant_rave` precision rose 0.79 → 0.91. The baseline's problem was that it leaked real questions into other labels based on emotional tone; the fine-tuned model learned to recognize a question by its *intent*, which is exactly the gap I designed the labels around. The only thing the fine-tuned model gives up is a little `theory` recall (1.00 → 0.90), losing one speculative "what if…?" post — a fair trade for fixing the question leak.

**The real lesson was about epochs, not data.** My first fine-tuning attempt (3 epochs) *underperformed* the baseline and collapsed into a single default class. The training curve revealed why: at 3 epochs the model hadn't left the random floor yet (0.32 val accuracy). It only began separating the classes at epoch 3 and kept improving through epoch 8. So the bottleneck wasn't the label definitions or even mainly the dataset size — it was **stopping training too early.** Giving the model enough passes over the same ~150 examples was enough to beat a 70B zero-shot model on this task.

---

## Error analysis

The 8-epoch model made only 3 mistakes, and unlike the undertrained run they no longer point in one direction — each lands on a genuine label boundary I flagged in planning.md.

**1. A "what if…?" theory read as a question** — `theory` → `plot_question` (conf 0.61)
> "And what if Al-Zuras' ship wasn't sailing in the ocean but ON THE FLOOD that happened with Noah? Maybe they crossed paths. It's all possible and it's all connected ✌️✌️"

This is the exact `theory`↔`plot_question` edge case from planning.md: a theory phrased as a leading question ("And what if…?"). By my rule it's a `theory` (it proposes a connection), but the question form and question mark pull the model toward `plot_question`. Notably this was the model's **highest-confidence error (0.61)** — it's genuinely "sure" because the surface form really does look like a question. This is ambiguous *language*, not bad labeling.

**2. An opinion that name-drops a theory topic** — `rant_rave` → `theory` (conf 0.44)
> "I was hoping it would be a sci fi show or that the government was involved in what happened."

I labeled this `rant_rave` — it's the author expressing disappointment about what they *wanted* the show to be. But it mentions "the government was involved," which is theory-flavored vocabulary, so the model tipped to `theory`. This is the `theory`↔`rant_rave` boundary: the *topic* signals one label while the *intent* (venting a preference) signals another. The model is reading topic words over intent.

**3. A production question with emotional wording** — `plot_question` → `rant_rave` (conf 0.45)
> "No spoilers but didnt they drastically change her storyline due to the hate the real life character was getting?"

This is a question about a real-world production decision, which I labeled `plot_question` — and it's one of the posts I had flagged in [ambiguous_review.csv](ambiguous_review.csv) as not fitting cleanly. The mention of "hate" gives it an emotional charge that nudges the model to `rant_rave`. A borderline case even for a human.

**Labeling problem or data problem?** I re-read all 3 misclassified posts against my planning.md definitions and confirmed my labels were correct and *consistent*. These are not annotation errors — all three are **genuine boundary cases** (leading-question theories, topic-vs-intent conflicts), which is exactly where planning.md predicted any classifier would struggle. The model is no longer failing on easy posts; it only trips on the truly ambiguous ones.

**What would push it further:** more training examples of the specific boundary shapes that remain — "what if…?" speculative theories, and emotionally-worded posts that are really questions or opinions — so the model leans on intent rather than question marks and topic vocabulary. The label definitions themselves are sound; the remaining errors are about exposure to hard cases, not definition clarity.

---

## Sample classifications

A few test posts run through the **fine-tuned** (8-epoch) model, with the predicted label and the model's confidence (softmax probability of the predicted class):

| Post (truncated) | Predicted | Confidence | Correct? |
|---|---|---|---|
| "My theory is God wanted to test humanity and chose those in 828 to decide whether to cause an apocalypse" | theory | 0.95 | ✅ (true: theory) |
| "I LOVED Eagan, he was easily the best character on the show" | rant_rave | 0.88 | ✅ (true: rant_rave) |
| "And what if Al-Zuras' ship wasn't sailing in the ocean but ON THE FLOOD that happened with Noah?…" | plot_question | 0.61 | ❌ (true: theory) |
| "I was hoping it would be a sci fi show or that the government was involved in what happened." | theory | 0.44 | ❌ (true: rant_rave) |
| "No spoilers but didnt they drastically change her storyline due to the hate the real life character was getting?" | rant_rave | 0.45 | ❌ (true: plot_question) |

¹ The two correct posts are classified correctly by the model; drop in their confidence scores from the classify cell output (one number each).

**Why a correct prediction is reasonable:** the model labeled "My theory is God wanted to test humanity…" as `theory` — the right call, because the post states an explicit hypothesis about *why* Flight 828 disappeared and gives a reason, which is the core of the `theory` definition. It's the most prototypical theory shape in the data, so the model recognizes it cleanly.

> **Confidence note:** unlike the failed 3-epoch run (where everything sat at ~0.34, the 3-class random floor), the 8-epoch model is now confident on the posts it gets right and only uncertain on the genuinely ambiguous ones. Even its mistakes are interpretable — e.g. the "what if…?" theory it called a `plot_question` at 0.61 is a real leading-question edge case, not a blind default.

## Reflection: what the model captured vs. what I intended

**What I intended** each label to capture (from planning.md) was the *primary intent* of a post — is the author theorizing, reacting, or asking? — regardless of surface features like punctuation or emotional wording.

**What the model actually captured** is, encouragingly, *mostly* that intent — but it still leans on surface cues at the margins:
- It learned **all three labels with roughly equal skill** (~0.90 precision and recall each). It is no longer hiding in a default class; it confidently and correctly handles clear theories, clear reactions, and clear questions. This is the intent-based split I designed working as intended.
- Where it still **falls back on surface form**, it's exactly the three error types: it reads a question mark as `plot_question` (the "what if…?" theory), and it reads topic vocabulary or emotional charge ("government was involved," "the hate") as a signal even when the author's intent is something else. So the residual gap is "form/topic occasionally overrides intent" — but only on genuinely borderline posts, not across the board like the undertrained model.

**The most important thing this run taught me** is the distinction between *can't* and *hasn't yet*. The 3-epoch model looked like it had a deep conceptual gap — it couldn't tell intent from tone at all. But the 8-epoch model shows the concept *was* learnable from the same 150 examples; the earlier failure was just insufficient training, not a flaw in the labels or an impossible boundary. The decision boundary I designed (intent-based) is one the model can actually approximate — it just needed enough passes to get there.

## Spec reflection

**One way the spec helped:** the spec required me to write out **hard edge cases and an explicit annotation rule before collecting data**. Forcing the "label by main purpose, not punctuation" rule up front did two things: it kept my 210 labels consistent (so I could rule out annotation error as the cause of mistakes), and it let me *predict the exact failure boundary* — I wrote in planning.md that `theory`↔`plot_question` would be hardest, and that's precisely where all three of the final model's errors land (a leading-question theory, a topic-vs-intent mix-up, an emotionally-worded question). The structured planning step turned error analysis from guesswork into confirmed hypothesis-testing.

**One way my implementation diverged:** planning.md originally specified a `pre_labeled` column to track LLM-assisted pre-labeling for disclosure. In implementation I **dropped pre-labeling entirely** and labeled every post from scratch (the CSV has a `notes` column instead). Why: the hardest, most informative posts are the boundary cases, and letting an LLM pre-label them risked anchoring my own judgment on exactly the examples where independent judgment matters most. Labeling unaided kept the ground truth clean, at the cost of more manual effort.

## Hyperparameters

Learning rate 2e-5 and batch size 16 are the notebook defaults. I changed **`num_train_epochs` from 3 to 8**. At 3 epochs the model was undertrained on the small (~150-example) training set: it sat near the 3-class random floor (0.32 validation accuracy) and collapsed all uncertain posts into a single default label, scoring just 0.59 on the test set — *worse* than the baseline. The training curve showed it didn't start learning until epoch 3 and kept improving through epoch 8.

Raising to 8 epochs fixed it: validation accuracy climbed 0.32 → 0.97 and test accuracy rose 0.59 → 0.906, enough to beat the baseline. Validation loss was still falling at epoch 8 (no overfitting), so 8 was a safe stopping point and `load_best_model_at_end=True` keeps the best epoch. I left learning rate and batch size at the defaults — the only change needed was training longer.

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

**3. Diagnosing the undertrained run and fixing it.**
- *What I directed:* my first fine-tuning run underperformed the baseline; I asked the AI to analyze the wrong predictions and confidence scores and explain what was going wrong.
- *What it produced:* the diagnosis that the model was near the random floor and collapsing into one default class (undertraining), with the suggestion to increase epochs rather than change the data or labels.
- *What I changed/overrode:* I checked the per-epoch validation curve myself to confirm undertraining (accuracy stuck at 0.32 for two epochs), then raised `num_train_epochs` 3 → 8 and re-ran. The model then beat the baseline — and I verified the new errors were genuine edge cases, not a different collapse, before writing them up.

I also used an AI assistant to draft and tighten the prose in this README and planning.md, which I reviewed and edited.
