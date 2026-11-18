## Chapter 8 — Distributions and Softmax

Softmax is the formula that turns a model's raw scores, which can be any numbers at all, into a proper probability distribution: all positive and adding up to exactly 1. It takes two lines to write and is used at the end of every LLM and inside every attention layer. By the end of this chapter you will compute it by hand.

### 8.1 What is a probability distribution?

A **probability distribution** is a complete list of possible outcomes, each with a probability, where the probabilities add up to 1. It answers "what might happen, and how likely is each?"

Some everyday distributions:

| Situation | Outcomes and probabilities | Sum |
| --- | --- | --- |
| Tomorrow's weather forecast | Sun 0.5, cloud 0.3, rain 0.2 | 1.0 |
| A fair die | 1 to 6, each 1/6 | 1.0 |
| How customers pay at a café | Card 0.75, phone 0.20, cash 0.05 | 1.0 |
| Next word after "See you" | tomorrow 0.4, soon 0.3, later 0.2, other 0.1 | 1.0 |

Think of a distribution as a pie cut into slices. Every slice is some outcome. The slices can be different sizes, but together they make exactly one whole pie. A slice cannot have negative size, and there is no pie left over.

An LLM's output is exactly such a pie, sliced into as many pieces as there are tokens in the vocabulary, often over 100,000 slices. Most slices are microscopic. A few are large.

### 8.2 The problem: raw scores are not probabilities

In Chapter 4 you saw how the last layer of an LLM produces a score for each vocabulary word by taking dot products. These raw scores are called **logits**. They can be any number: positive, negative, large or small. For "The cat sat on the \_\_\_" a model might produce:

| Word | Logit |
| --- | --- |
| mat | 3.0 |
| floor | 2.0 |
| sofa | 1.5 |
| roof | 0.5 |
| banana | −1.0 |

Higher means "more fitting", but these are not probabilities: one is negative and they add up to 6.0, not 1. We need a recipe that:

1. Makes every number positive.
2. Makes them add up to 1.
3. Keeps the order: a higher score must give a higher probability.

**Attempt 1: divide each by the total.** 3.0 / 6.0 = 0.5 for "mat", fine. But "banana" gets −1.0 / 6.0 ≈ −0.17, a negative probability. Impossible. And if the scores summed to zero, we would divide by zero. This attempt fails.

**Attempt 2: first make them positive, then divide by the total.** What function turns any number into a positive one while keeping the order? Look back at Chapter 2, Figure 2.2: the exponential eˣ. It is always positive, and bigger inputs always give bigger outputs. That is exactly what we need.

### 8.3 Softmax, step by step

Softmax is Attempt 2:

1. Raise e to the power of each score.
2. Add up all those results.
3. Divide each result by the sum.

$$
\text{softmax}(z_i) = \frac{e^{z_i}}{\sum_{j} e^{z_j}}
$$

In words: the probability of option *i* is e to its score, divided by the sum of e to every option's score.

Let's do it for our five words:

| Word | Logit z | eᶻ | eᶻ ÷ 33.98 | Probability |
| --- | --- | --- | --- | --- |
| mat | 3.0 | 20.09 | 0.591 | 59.1% |
| floor | 2.0 | 7.39 | 0.217 | 21.7% |
| sofa | 1.5 | 4.48 | 0.132 | 13.2% |
| roof | 0.5 | 1.65 | 0.049 | 4.9% |
| banana | −1.0 | 0.37 | 0.011 | 1.1% |
| **Total** |  | **33.98** | **1.000** | **100%** |

All three requirements are met: every probability is positive (even banana's), they add up to 1, and the order is preserved.

![Figure 8.1 · Five illustrative scores; probabilities computed from them](figures/fig-08-01.png)

> **Try it yourself.** Compute the softmax of the scores \[1, 1, 1\]. (Every e¹ is the same, so each gets 1/3.) Now try \[2, 1, 0\]. You should get about \[0.67, 0.24, 0.09\].

### 8.4 What softmax does to differences

Look again at Figure 8.1. On the left, "mat" scores 3.0 and "floor" 2.0, a difference of 1. On the right, "mat" gets 59% and "floor" 22%, nearly three times as much. Why?

Because eˣ turns **differences into ratios**. A difference of 1 in score always means a ratio of e ≈ 2.72 in probability. A difference of 2 means e² ≈ 7.4 times. A difference of 4 means e⁴ ≈ 55 times.

$$
\frac{P(\text{mat})}{P(\text{floor})} = \frac{e^{3.0}}{e^{2.0}} = e^{3.0 - 2.0} = e^{1} \approx 2.72
$$

This gives softmax two important properties:

**Only differences matter.** Add 100 to every score and the probabilities do not change at all, because every ratio stays the same. Scores like \[103, 102, 101.5, 100.5, 99\] give exactly the same distribution as \[3, 2, 1.5, 0.5, −1\]. The model only needs to get the *relative* scores right.

**It is a "soft" maximum.** If one score is far above the rest, softmax gives it nearly all the probability, like picking the maximum. If scores are close, it shares probability among them. That is where the name comes from: a hard max would put 100% on the winner; softmax gives the winner the most, softly.

> **Real-world analogy.** Imagine a talent show where the prize money is split by applause. With a "soft" rule, the loudest act gets the most but others get something. With a "hard" rule, the winner takes all. Softmax is the soft rule, and temperature (next) controls how soft.

### 8.5 Temperature: turning the dial between bold and cautious

You may have seen a "temperature" setting in AI tools. It is a single number that divides every score before softmax:

$$
P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

![Figure 8.2 · The same five scores at T = 0.5, 1 and 2](figures/fig-08-02.png)

| Temperature | What dividing by T does | Effect on the distribution | Typical use |
| --- | --- | --- | --- |
| T < 1 (e.g. 0.5) | Stretches the scores apart | Sharper: the top choice dominates (84% for "mat") | Factual answers, code |
| T = 1 | No change | The model's natural distribution | Default |
| T > 1 (e.g. 2) | Squeezes the scores together | Flatter: more variety ("mat" drops to 40%) | Brainstorming, creative writing |
| T → 0 | Infinitely stretched | All probability on the top choice | Deterministic output |

The name comes from physics: at high temperature, molecules move randomly in all directions; at low temperature, they settle into the most stable state. Chapter 17 returns to temperature when we look at how text is actually generated.

### 8.6 Softmax inside attention

Softmax appears in a second, equally important place. In attention (Chapter 14), each word computes a relevance score for every other word, using dot products. Softmax turns those scores into **weights** that are positive and sum to 1, so they can be used to make a weighted mixture, exactly like the 70% / 30% smoothie in Chapter 3.

So softmax plays two roles in an LLM:

| Where | Input | Output |
| --- | --- | --- |
| End of the model | One logit per vocabulary token | Probability of each next token |
| Inside every attention layer | One relevance score per other token | Mixing weights for combining those tokens' information |

### 8.7 A practical detail: overflow

Computers store numbers with limited range. e₁₀₀₀ is far too large to store, so if a logit happened to be 1,000, a naive softmax would crash or produce nonsense. Because only differences matter (8.4), the fix is easy: subtract the largest score from every score first. The largest becomes 0 (and e⁰ = 1), every other score becomes negative, and nothing overflows. The probabilities come out exactly the same.

> **Developer note.** A safe softmax in NumPy:
>
> ```python
> def softmax(z, T=1.0):
>     z = np.asarray(z) / T
>     z = z - z.max()          # avoid overflow; result unchanged
>     e = np.exp(z)
>     return e / e.sum()
> softmax([3, 2, 1.5, 0.5, -1])   # [0.591, 0.217, 0.132, 0.049, 0.011]
> ```

### 8.8 Describing a distribution

Two questions summarize any distribution:

1. **What is the most likely outcome?** For our example, "mat" at 59%. This is called the **mode**.
2. **How spread out is it?** Is the model nearly certain (one huge slice) or torn between many options (lots of similar slices)?

Compare two forecasters. One says "rain 95%". The other says "sun 35%, cloud 35%, rain 30%". The first is confident; the second is unsure. Being able to measure "how unsure" with a single number is useful: it tells you when a model is guessing. That measure is called **entropy**, and it is the first topic of Chapter 9.

### Chapter summary

- A probability distribution lists every outcome with a probability; probabilities are non-negative and sum to 1.
- Raw model scores are called logits and can be any real number.
- Softmax exponentiates each logit and divides by the total, producing a valid distribution.
- Softmax turns score differences into probability ratios; adding a constant to all scores changes nothing.
- Temperature divides scores before softmax: low T sharpens, high T flattens.
- Softmax is used for next-token probabilities and for attention weights.

> **Try it yourself.**
>
> 1. Softmax the scores \[0, 0\]. Then \[5, 0\]. How much of the probability does the first option get in each case?
> 2. Two words have logits 4 and 2. What is the ratio of their probabilities?
> 3. If you double the temperature, does the top word's probability go up or down?
