# TakeMeter — Planning Document

**AI201 · Project 3**
**Community:** the *Manifest* television-series fandom

**Task:** classify fan posts into how the viewer is *engaging* with the show

---

## Community

I chose the ***Manifest* fandom** — the community of viewers who follow the NBC/Netflix mystery-drama about the passengers of Flight 828, who vanish and return 5½ years later with supernatural "Callings."

This community is a good fit for a classification task because the show is built on an unresolved, serialized mystery, so the discourse splits cleanly into unique styles of engagement rather than just distinct topics. 

The same episode produces (1) rigorous, evidence-based puzzle-solving, (2) raw emotional reactions to characters and relationships, and (3) confused viewers trying to catch up on the timeline. Those three modes look very different in tone, structure, and intent, which gives a classifier real signal to learn, but they also overlap at the edges (a theory can be emotional; a question can hint at a theory), which makes the problem genuinely interesting rather than trivially separable.

---

## Labels

I use **3 labels**, mirroring the `LABEL_MAP` in the notebook.

### `theory`
A post that lays out a **structured hypothesis or prediction** about the show's mysteries, supporting it with reasoning, evidence, or references to specific clues.
- *Example 1:* "I think the Callings are the passengers' own future memories leaking backward — every Calling we've seen has later mapped to something the Death Date forces them to do. The driftwood and the sapphire are the mechanism."
- *Example 2:* "Prediction: Angelina isn't the antagonist, she's a mirror of Cal. The sapphire amplifies intent, so the two of them are the 'two halves' the mythology keeps hinting at, and the finale will force them to choose together."

### `rant_rave`
A post that is primarily an **emotional reaction** — praising or criticizing a character, relationship, or plot point — with little or no analytical claim.
- *Example 1:* "I cannot STAND how they wrote Michaela this season. Every decision she makes is infuriating and the Zeke storyline broke my heart for nothing."
- *Example 2:* "Ben and Saanvi carrying this entire show on their backs 😭 the scene in the lab was everything, best episode they've ever done, I'm obsessed."

### `plot_question`
A **factual question** about something the viewer missed or found confusing in the timeline, characters, or events — seeking an answer, not offering a theory.
- *Example 1:* "Wait, how long was Cal actually gone before he came back older? I lost track of the timeline after the season 3 jump."
- *Example 2:* "Who was the guy in the tattoo parlor again, and is he the same person Michaela saw at the end of episode 4? I'm confused."

---

## Hard edge cases

The genuinely ambiguous posts sit **between `theory` and `plot_question`** — a "leading question" that is really a hypothesis in disguise, e.g. *"Isn't it weird that the Callings stopped right when the sapphire appeared? Almost like they're connected, right?"* This has a question mark (looks like `plot_question`) but is actually advancing a claim (`theory`).

**Rule for annotation:** if the post's main purpose is to **propose or defend an explanation**, label it `theory`, even if it's phrased as a question. Reserve `plot_question` for posts that are genuinely seeking a fact the asker doesn't know and isn't speculating about. A secondary edge case is **`rant_rave` vs `theory`** (an emotional post that also makes a claim) — label by the dominant intent: if removing the emotion leaves a real argument, it's `theory`; if removing the emotion leaves nothing, it's `rant_rave`.

---

## Data collection plan

**Sources (in priority order):**
- **r/Manifest (Reddit)** — primary source; richest mix of all three labels in episode-discussion and theory threads.
- **X/Twitter (#Manifest)** — short, high-emotion posts; strongest source for `rant_rave`.
- **Netflix / IMDb episode discussions & review comments** — episode recaps and confused-viewer comments; good for `plot_question`.
- **Facebook fan groups** — casual viewers; supplements `plot_question` and `rant_rave`.

**Target volume:** **200+ examples total, ~67 per label** to keep classes balanced for stratified splitting.

**If a label is underrepresented after 200 examples:** `theory` is the easiest to over-collect and `plot_question` the hardest, so if `plot_question` lags I'll **targeted-collect** from IMDb/Netflix episode-comment threads and Reddit "Question" flairs/megathreads, which are dense with factual questions. If a label still can't reach a usable floor (~50), I'll down-sample the over-represented classes so the training set stays roughly balanced and note the final per-class counts in the README.

---

## Evaluation metrics

Accuracy alone is not enough here because the classes may be mildly imbalanced and, more importantly, because the cost of different confusions is not uniform (mislabeling a `theory` as a `plot_question` matters more than a minor `rant_rave` slip). I will use:

- **Macro-averaged F1** as the headline metric — it weights each class equally, so a model that nails the easy `rant_rave` posts but fails on the rarer/harder `theory`↔`plot_question` boundary is correctly penalized.
- **Per-class precision and recall** — to see *which* label is failing and in which direction (e.g. low `theory` recall = theories leaking into other classes).
- **Confusion matrix** — to confirm whether errors concentrate on the predicted hard edge (`theory` ↔ `plot_question`), which validates whether my label definitions are doing their job.

I'll report all three for both the fine-tuned model and the Groq baseline.

---

## Definition of success

For this tool to be **genuinely useful** — e.g. auto-tagging posts in the subreddit so theory-hunters can filter to `theory` and mods can route `plot_question` posts to a help thread — it needs to be reliable enough that humans only spot-correct rather than re-label everything.

- **Good enough for deployment:** **~0.80 macro-F1**, with no single class below ~0.70 recall. At that level the tool can auto-tag with light human review and still save real effort.
- **Baseline expectation:** the fine-tuned DistilBERT should **beat the zero-shot Groq baseline** on macro-F1; if it doesn't, that's a finding worth analyzing (likely too few examples on the hard boundary).

---

## Label stress-testing (pre-annotation)

Before annotating, I'll give an LLM my three definitions **and the `theory`↔`plot_question` edge case**, and ask it to generate 5–10 posts that sit *on* that boundary (leading questions, emotional theories). If I can't cleanly classify its output using my "dominant intent" rule, my definitions are too loose and I'll tighten them — most likely by adding the explicit "phrased as a question but proposing an explanation → `theory`" clause (already reflected above) before touching the real 200.

## Annotation assistance

I **will** use an LLM to **pre-label** batches before reviewing every example myself; I am the final annotator on 100% of examples. To track this for the AI-usage disclosure, my CSV will include a `pre_labeled` column (model name + whether I changed its label), so I can report how many pre-labels I accepted vs. corrected. No example enters the dataset without my manual confirmation.

## Failure analysis (post-evaluation)

After evaluation, I'll take the notebook's list of wrong predictions (Section 4 prints them with true/predicted/confidence) and ask an AI tool to **identify error patterns** — e.g. "are most errors `theory` predicted as `plot_question`?", "do errors cluster on short posts or low-confidence cases?". I will then **verify each claimed pattern myself** by reading the actual misclassified posts before writing it into the README, so the write-up reflects real, checked patterns rather than the model's unverified summary.
