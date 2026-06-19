# Demo Video Script — TakeMeter

Just read this top to bottom. Each part says what to show on screen and what to say.

---

## Before you record — the cell to run on camera

Add this cell after training, so labels and confidence show on screen:

```python
import torch, torch.nn.functional as F

def classify(text):
    model.eval()
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(model.device)
    with torch.no_grad():
        probs = F.softmax(model(**inputs).logits, dim=-1)[0]
    pred = int(probs.argmax())
    print(f"POST: {text}")
    print(f"  -> PREDICTED: {ID_TO_LABEL[pred]}  (confidence {probs[pred]:.2f})")
    for i, p in enumerate(probs):
        print(f"       {ID_TO_LABEL[i]:14s} {p:.2f}")
    print("-" * 70)

demo_posts = [
    "I LOVED Eagan, he was easily the best character on the show",                                   # rant_rave (correct)
    "Why did Cal age when he was gone?",                                                             # plot_question (correct)
    "My theory is God wanted to test humanity and chose those in 828 to decide whether to cause an apocalypse",  # theory (correct)
    "And what if Al-Zuras' ship wasn't sailing in the ocean but ON THE FLOOD that happened with Noah? It's all connected",  # theory -> model says plot_question (WRONG)
]
for p in demo_posts:
    classify(p)
```

---

## Part 1 — Intro (about 30 seconds)

**Show:** your Colab notebook.

**Say:**
> "Hi, this is my project called TakeMeter. It's a classifier for the *Manifest* TV show fandom on Reddit. It looks at a fan's post and sorts it into one of three types: a **theory**, which is a guess about the show's mysteries; a **rant or rave**, which is just an emotional reaction; or a **plot question**, which is someone asking about something they missed. I trained a model called DistilBERT on 210 Reddit posts, and I compared it against a zero-shot baseline from Groq."

---

## Part 2 — Show it classifying posts (about 1 minute)

**Show:** run the `classify` cell so the posts, labels, and confidence numbers print on screen.

**Say:**
> "Here I'm running four real posts through my fine-tuned model. For each one it prints the predicted label and how confident it is, from 0 to 1. You can see the label and the confidence score right here for each post."

*(Give it a second so the screen is readable.)*

---

## Part 3 — One it got RIGHT (about 45 seconds)

**Show:** point to *"My theory is God wanted to test humanity…"* (or whichever clearly-correct one shows a confident label).

**Say:**
> "Let's look at one it got right. This post says 'My theory is God wanted to test humanity…' and the model labeled it **theory**, with good confidence. That makes sense — the post literally states a hypothesis about why the disappearance happened and gives a reason. That's exactly what a theory is. And notice the confidence is solid here, not a random guess — the model has clearly learned what a theory looks like."

---

## Part 4 — One it got WRONG (about 45 seconds)

**Show:** point to *"And what if Al-Zuras' ship wasn't sailing in the ocean but ON THE FLOOD…"*

**Say:**
> "Now here's one it got wrong. This post says 'And what if Al-Zuras' ship was on the flood from Noah… it's all connected.' I labeled this a **theory**, because the person is proposing a connection between the show and the Noah's Ark story. But the model called it a **plot question**, because it's phrased as a 'what if' question with a question mark. So the model got fooled by the question form. This is actually the exact hard edge case I predicted in my planning doc — a theory that's written like a question. It's a genuinely tricky one, not a careless mistake."

---

## Part 5 — Walk through the evaluation report (about 1 minute 15 seconds)

**Show:** switch to your README on GitHub.

**Say:**
> "Now let's look at my evaluation report. At the top, my fine-tuned model scored about 0.91, and the baseline scored 0.87. So the fine-tuned model **won** — but there's a story behind that."

**Show:** scroll to the training curve table.

**Say:**
> "At first my fine-tuned model actually did worse than the baseline. This training curve shows why. For the first two epochs the model was stuck at 0.32 accuracy, which is basically random guessing between three labels. It was undertrained. Once I increased the training from 3 epochs to 8, you can see the accuracy climb all the way up to 0.97 on the validation set. So the fix wasn't more data or different labels — it just needed to train longer."

**Show:** scroll to the confusion matrix table.

**Say:**
> "Here's the confusion matrix for the final model. It only made 3 mistakes, and they're spread out — one in each class, in three different directions. There's no single label it dumps everything into anymore. Every label gets about 90 percent precision and recall, which is balanced and healthy."

**Show:** scroll to the Error analysis.

**Say:**
> "And when I looked at those 3 mistakes, they were all genuine edge cases — like the 'what if' theory I just showed you, and a post where the topic sounded like a theory but the person was really just venting. These are exactly the boundaries I flagged as hard in my planning doc. So the model isn't failing on easy posts anymore, only on the truly ambiguous ones."

---

## Part 6 — Wrap up (about 30 seconds)

**Say:**
> "So the big takeaway: I defined my labels by the person's *intent* — are they theorizing, reacting, or asking? My first run made it look like the model couldn't learn that. But after training longer, it learned all three labels well and beat a much bigger zero-shot model. The lesson for me was the difference between a model that *can't* learn something and one that just *hasn't trained enough yet*. Thanks for watching!"

---

### Quick reminders before you record
- Run the whole notebook first so the model is loaded, then run the classify cell once to check the output looks right.
- Have two tabs open: Colab (for the live demo) and the GitHub README (for the report).
- Only explain one correct and one wrong post in detail — don't go through all four.
- The Al-Zuras "what if" post is your reliable wrong example (it's one of the model's 3 real errors). If for some reason it shows up correct on screen, use either of the other two errors instead: "I was hoping it would be a sci fi show or that the government was involved" (true rant_rave, model says theory) or "No spoilers but didnt they drastically change her storyline due to the hate…" (true plot_question, model says rant_rave).
