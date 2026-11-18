## Chapter 17 — Generating Text: From Probabilities to Words

A trained model only ever produces one thing: a probability distribution for the next token. Turning that into a paragraph means picking one token, adding it to the text, and asking again, over and over. *How* you pick, always the top choice or a weighted random draw, and how you trim the options first, changes the output from robotic to creative. This chapter covers those choices.

### 17.1 One token at a time

![Figure 17.1 · The autoregressive generation loop](figures/fig-17-01.png)

This is called **autoregressive generation**: each new token is generated from ("regressed on") all the tokens before it, including the ones the model itself just wrote. It is the chain rule of probability (Chapter 7) run forwards: P(w₁) then P(w₂ | w₁) then P(w₃ | w₁, w₂), choosing a word at each step instead of just scoring one.

Two consequences follow:

- **Answers stream word by word** because they really are produced word by word.
- **Early choices commit the model.** Once it has written "The answer is no, because", every later token must make sense after that. A model cannot go back and revise a token it has already emitted (though it can write "Actually, I made a mistake" and continue from there).

### 17.2 Greedy decoding: always take the top choice

The simplest strategy is **greedy decoding**: at each step, pick the single most likely token. It is predictable: the same prompt always gives the same output.

Greedy decoding is good when there is one right answer ("2 + 2 ="), but has two weaknesses:

- **Bland text.** Always choosing the most common continuation produces safe, generic sentences.
- **Loops.** Greedy decoding can fall into repetition: "I like the movie. I like the movie. I like the movie." Once a phrase has appeared, repeating it can become the most likely continuation.

It is like a restaurant guest who always orders the most popular dish: never a bad meal, never a memorable one.

### 17.3 Sampling: rolling a weighted die

**Sampling** picks the next token at random, but in proportion to its probability. A token with 59% probability is chosen 59% of the time, one with 1% is chosen 1% of the time.

Here is how a computer does it. Line up the tokens and give each a slice of the number line from 0 to 1, as wide as its probability:

| Token | Probability | Slice of the number line |
| --- | --- | --- |
| mat | 0.59 | 0.00 to 0.59 |
| floor | 0.22 | 0.59 to 0.81 |
| sofa | 0.13 | 0.81 to 0.94 |
| roof | 0.05 | 0.94 to 0.99 |
| banana | 0.01 | 0.99 to 1.00 |

Now draw a random number between 0 and 1. If it is 0.42, the token is "mat". If it is 0.87, "sofa". If it is 0.995, "banana". Over many draws, each token wins exactly as often as its probability says. It is a spinning prize wheel with slices of different widths.

Sampling gives variety, but the long tail is risky. A vocabulary has 100,000 tokens. Even if each nonsense token has only 0.001% probability, together they may add up to several percent, so across a 500-token answer, some nonsense is likely to slip in. The next three sections are ways to control this.

### 17.4 Temperature, revisited

Chapter 8 introduced **temperature**: dividing the logits by T before softmax. With the same scores as before:

| Token | T = 0.5 | T = 1 | T = 2 |
| --- | --- | --- | --- |
| mat | 84% | 59% | 40% |
| floor | 11% | 22% | 24% |
| sofa | 4% | 13% | 19% |
| roof | 0.6% | 5% | 11% |
| banana | 0.03% | 1% | 5% |

Low temperature concentrates the probability on the leaders, so sampling behaves almost like greedy decoding. High temperature spreads it out, giving more surprising, and more error-prone, choices. At T = 0, sampling becomes exactly greedy.

### 17.5 Top-k and top-p: trimming the tail

Temperature reshapes the whole distribution. A different idea is to **cut off the unlikely tail entirely** before sampling.

**Top-k sampling** keeps only the k most likely tokens (say k = 40), sets every other probability to zero, rescales the survivors so they add up to 1 again, and samples from them.

The problem with top-k is that a fixed k does not fit every situation. After "The capital of France is", only one token is sensible, so even k = 3 lets in two bad options. After "My favourite season is", several tokens are fine.

**Top-p sampling** (also called **nucleus sampling**) adapts: sort the tokens by probability and keep the smallest set whose probabilities add up to at least p (say 0.9). When the model is confident, that set is tiny; when it is unsure, the set grows.

![Figure 17.2 · Illustrative probabilities; cut-offs computed from them](figures/fig-17-02.png)

In Figure 17.2, the running total passes 90% at the fifth token, so top-p = 0.9 keeps autumn, summer, winter, spring and fall, and drops the long tail. A fixed top-k of 3 would have thrown away "spring", a perfectly good answer. After "The capital of France is", with "Paris" at 97%, the same top-p would keep just one token.

The kept probabilities are rescaled to add up to 1. For example, if the five kept tokens add to 0.90, then "autumn" becomes 0.28 / 0.90 ≈ 0.31.

### 17.6 Other controls

| Setting | What it does | Analogy |
| --- | --- | --- |
| Max tokens | Stops after a set length | A word limit on an essay |
| Stop sequences | Stops when a chosen string appears | "Over and out" on a radio |
| Repetition or frequency penalty | Lowers the logits of tokens already used | A teacher circling repeated words |
| Seed | Fixes the random numbers so sampling is repeatable | Shuffling a deck the same way every time |

### 17.7 Why answers vary, and why models sometimes invent things

With sampling, the same prompt can give different answers each time, because different random draws pick different tokens, and each choice changes everything after it. That variety is a feature for brainstorming and a nuisance when you want consistency.

The generation process also explains a well-known weakness: **hallucination**, confidently stated content that is false. The model always produces *some* next token, chosen because it is plausible, not because it was checked against reality. If the model's knowledge of a fact is weak, the distribution may still favour fluent, specific-sounding tokens: a plausible date, a plausible-looking citation. And once an invented detail is written, the following tokens build on it. Fluency and confidence in the text are not evidence of accuracy. Modern systems reduce this with training (Chapter 16) and by giving the model real sources to read (retrieval, Chapter 4), but it is wise to verify important facts.

### 17.8 How fast is generation?

Each new token requires one full forward pass through the model. Prompts are fast to read, because all their tokens can be processed in parallel. Answers are slower, because tokens must be produced one after another.

The main speed-ups:

- **KV cache** (Chapter 14): store earlier keys and values so each new token needs only a small amount of new computation.
- **Batching:** serve many users at once, so the GPU does many matrix multiplications together.
- **Speculative decoding:** a small, fast model drafts several tokens ahead; the big model checks them all in one pass and keeps the ones it agrees with. Like a junior writer drafting and a senior editor approving several words at a time.
- **Quantization** (Chapter 18): store weights with fewer bits, so they move through the chip faster.

### 17.9 Choosing settings in practice

| Task | Typical temperature | Typical top-p | Why |
| --- | --- | --- | --- |
| Code, data extraction, maths | 0 to 0.3 | 1.0 or lower | You want the single most likely correct answer |
| Factual Q&A, summaries | 0.3 to 0.7 | 0.9 | Mostly precise, a little natural variety |
| Conversation | 0.7 to 1.0 | 0.9 to 0.95 | Natural and varied |
| Brainstorming, fiction | 0.9 to 1.2 | 0.95 | Surprise is the point |

These are starting points, not rules. Different models respond differently, and many products choose these settings for you.

> **Developer note.** A complete sampler with temperature and top-p:
>
> ```python
> def sample(logits, T=0.8, top_p=0.9, rng=np.random.default_rng()):
>     p = softmax(np.asarray(logits) / max(T, 1e-6))
>     order = np.argsort(p)[::-1]
>     keep = order[: np.searchsorted(np.cumsum(p[order]), top_p) + 1]
>     q = p[keep] / p[keep].sum()
>     return rng.choice(keep, p=q)
> ```

### Chapter summary

- LLMs generate autoregressively: predict, pick one token, append, repeat.
- Greedy decoding always takes the top token: predictable but bland and loop-prone.
- Sampling draws tokens in proportion to their probability: varied but exposed to the long tail.
- Temperature sharpens or flattens the distribution; top-k and top-p cut off the unlikely tail before sampling.
- Plausible-but-unchecked token choices explain hallucinations; verify important facts.
- Each token needs a forward pass; caching, batching and speculative decoding make generation fast.

> **Try it yourself.**
>
> 1. Using the number-line table in 17.3, which token is chosen for random numbers 0.10, 0.60 and 0.97?
> 2. With probabilities \[0.5, 0.3, 0.1, 0.05, 0.05\] and top-p = 0.8, which tokens survive? What are their rescaled probabilities?
> 3. Why might a coding assistant use temperature 0 but a poetry assistant use 1.0?
