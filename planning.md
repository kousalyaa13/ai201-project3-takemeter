# TakeMeter — Planning Document

**AI201 · Project 3**
**Community:** the *Manifest* TV fandom
**Task:** classify fan posts by how the viewer is engaging with the show

---

## Community

I chose the *Manifest* fandom — fans of the NBC/Netflix mystery show about the passengers of Flight 828, who disappear and come back 5½ years later with visions called "Callings."

This is a good community for a classification task because the whole show is one ongoing mystery, so fans engage in clearly different ways: some build detailed theories, some just react emotionally to characters, and some ask questions about things they missed. The three styles look different enough for a model to learn, but they overlap at the edges, which makes the task interesting instead of trivial.

---

## Labels

I use 3 labels, matching the `LABEL_MAP` in the notebook.

### `theory`
A post that lays out a hypothesis or prediction about the show's mysteries.
- "My theory is God wanted to test humanity and chose those in 828 as a sample set to decide whether to cause an apocalypse"
- "the tail fin teleported to the ocean because Saanvi killed the Major."

### `rant_rave`
A post that is mainly an emotional reaction — liking or disliking a character, relationship, or plot point.
- "I LOVED Eagan, he was easily the best character on the show"
- "This show is so freaking bland and Grace is so annoying that I couldn't even finish season 2."

### `plot_question`
A factual question about something the viewer missed or found confusing.
- "Why did Cal age when he was gone?"
- "Who shot Grace? I rewatched and still can't tell who was holding the gun."

---

## Hard edge cases

Some posts sit between two labels. My rule is to label by the post's **main purpose**, not its punctuation.

1. **`theory` vs `plot_question` — a question that is really a theory.**
   *"How did the Major know that the passengers' minds were connected as soon as they landed? ... This suggests that the government was involved."*
   It has a question mark but it's arguing a point. → If the post is making a claim, label `theory`; if it just wants a fact, label `plot_question`.

2. **`rant_rave` vs `plot_question` — a complaint shaped like a question.**
   *"Why is the casting on this show so terrible?"*
   It isn't really asking for information, it's venting. → `rant_rave`.

3. **`theory` vs `rant_rave` — an opinion that also makes a claim.**
   *"I believe Jared kinda wanted to have his cake and eat it too: be with Lourdes but still love Michaela."*
   → If removing the emotion still leaves a real argument, label `theory`; if it leaves nothing, label `rant_rave`.

When I hit one of these while labeling, I wrote the reason in a `notes` column so I'd stay consistent.

---

## Data collection plan

All examples come from **Reddit** (r/Manifest and related Manifest threads).

I aimed for 200+ examples, roughly balanced across the three labels. If a label was running short, I searched question and theory threads specifically to fill it. Final counts: `plot_question` 72, `rant_rave` 70, `theory` 68 (210 total).

---

## Evaluation metrics

Accuracy alone isn't enough, because the classes aren't perfectly balanced and some mistakes matter more than others (mixing up `theory` and `plot_question` is worse than a small `rant_rave` slip). I'll use:

- **Macro-F1** as the main metric — it treats all three classes equally.
- **Per-class precision and recall** — shows which label is failing and in which direction.
- **Confusion matrix** — shows whether errors cluster on the `theory` ↔ `plot_question` boundary.

I'll report these for both the fine-tuned model and the Groq baseline.

---

## Definition of success

This tool would be useful for auto-tagging posts — for example, so fans can filter to theories or route confused viewers to a help thread. To trust it with only light human review, I'd want about **0.80 macro-F1**, with no class below ~0.70 recall. I also expect the fine-tuned model to beat the zero-shot Groq baseline; if it doesn't, that's a result worth analyzing.

---

## Label stress-testing

Before labeling everything, I gave an AI my definitions and edge cases and asked it to write posts that sit on the `theory` ↔ `plot_question` boundary. The posts I couldn't label cleanly showed me where to tighten my rules — that's where the "label by main purpose" rule came from.

## Annotation assistance

I labeled every example myself. For disclosure: I used an AI assistant to help clean up CSV formatting and to flag borderline posts for me to re-check, but I made the final label decision on every post.

## Failure analysis

After evaluation, I'll take the notebook's list of wrong predictions and ask an AI to spot patterns (e.g. "are most errors `theory` predicted as `plot_question`?"). Then I'll check those patterns against the actual posts myself before writing them into the README.
