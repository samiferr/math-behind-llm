## Chapter 9 — Surprise, Entropy and Loss: How Wrong Is the Model?

To learn, a model needs a single number that says how wrong its predictions were. For LLMs that number is the **cross-entropy loss**: the average surprise the model felt at each real next token. This chapter builds it from one simple idea, surprise, and shows why it is such a good measure of mistakes.

### 9.1 Surprise: rare events carry more information

Compare two news headlines:

- "Sun rises in Toronto this morning."
- "Snowstorm hits Toronto in July."

The first tells you nothing; you were certain already. The second is shocking and genuinely informative. The less likely an event, the more you learn when it happens.

The scientist Claude Shannon turned this into a formula in 1948. The **surprise** (also called the information content) of an event with probability p is:

$$
\text{surprise}(p) = -\log_2(p) \quad \text{bits}
$$

Why a log? Because surprise should add up when independent events combine. Two coin flips both landing heads has probability 0.25, and it should be twice as surprising as one head. Logs turn multiplication into addition (Chapter 2): −log₂(0.25) = 2 = 1 + 1. Why the minus sign? Logs of numbers below 1 are negative, and we want surprise to be positive.

![Figure 9.1 · −log₂(p) for p from 0.01 to 1, values rounded](figures/fig-09-01.png)

| Event | Probability | Surprise |
| --- | --- | --- |
| Something certain | 1 | 0 bits |
| A coin landing heads | 1/2 | 1 bit |
| Rolling a 6 | 1/6 | 2.58 bits |
| A 1-in-1,000 event | 0.001 | 9.97 bits |

One **bit** is the surprise of a fair coin flip: the information in one yes/no answer. If you use the natural log (ln) instead of log₂, the unit is called a **nat**. Machine learning usually uses nats; the ideas are identical, only the scale differs (1 nat ≈ 1.44 bits).

### 9.2 Entropy: average surprise

A distribution can be more or less uncertain. **Entropy** measures this: it is the average surprise you expect, weighting each outcome's surprise by how likely it is.

$$
H(P) = -\sum_{x} P(x) \log_2 P(x)
$$

![Figure 9.2 · Entropy computed from each distribution's probabilities](figures/fig-09-02.png)

Three patterns are worth remembering:

- **Certainty has zero entropy.** If one outcome has probability 1, nothing is ever surprising.
- **Equal odds have maximum entropy.** Four equally likely options give 2 bits, eight give 3 bits. In general, n equal options give log₂(n) bits.
- **Entropy is the number of yes/no questions you need.** With 4 equally likely options, you need exactly 2 questions ("Is it A or B?" then "Is it A?"). This is the game of Twenty Questions: a good player asks questions that split the remaining options in half.

> **Real-world example.** Morse code gives the most common English letter, E, the shortest code (a single dot) and rare letters like Q long codes. That is entropy in action: frequent symbols should cost few bits, rare ones many. File compression (ZIP, MP3, JPEG) is built on the same principle.

> **LLM connection.** The entropy of a model's next-token distribution tells you how sure it is. After "The capital of France is", entropy is near zero: "Paris" takes almost all the probability. After "My favourite colour is", entropy is high: many answers are reasonable. Some systems watch entropy to detect when a model is guessing.

### 9.3 Cross-entropy: surprise under the wrong beliefs

Now the key step. Suppose the world follows one distribution (call it P, the truth), but you believe a different one (call it Q, your model). Every time something happens, you feel surprise based on *your* beliefs, −log Q(x). Averaged over what *actually* happens, this is the **cross-entropy**:

$$
H(P, Q) = -\sum_x P(x) \log Q(x)
$$

An analogy: two weather forecasters predict tomorrow. Forecaster A says "rain 80%", forecaster B says "rain 10%". It rains.

- A's surprise: −ln(0.8) ≈ 0.22
- B's surprise: −ln(0.1) ≈ 2.30

Over hundreds of days, the forecaster with lower average surprise is the better forecaster. Cross-entropy is exactly that average. It is lowest, and equal to the plain entropy of the truth, when your beliefs match reality.

### 9.4 The loss of a language model

For language, "the truth" at each position is simple: the word that actually came next in the training text. All the probability is on that one token. The cross-entropy formula then collapses to a single term:

$$
\text{loss} = -\ln Q(\text{correct token})
$$

That is the entire loss for one prediction: **the negative log of the probability the model gave the right answer.** For a whole sentence, average it over every position.

![Figure 9.3 · Illustrative probabilities for one sentence](figures/fig-09-03.png)

Reading Figure 9.3:

- "the" (second time) was easy: after "sat on", the model gave it 80%, so the loss is only 0.22.
- "cat" was hard: after "The", thousands of nouns are possible, and the model gave "cat" just 1%. Loss 4.61.
- The average loss for the sentence is about 1.89. Training means changing the model's parameters so this average goes down across billions of sentences.

Notice how this connects to the chain rule of probability (Chapter 7). The probability of the whole sentence is the product of the six probabilities. Its log is the sum of the six log-probabilities. So minimizing the average loss is the same as making the real training sentences as probable as possible. Statisticians call this **maximum likelihood**.

### 9.5 Why this loss is so good

Why not simply measure "right or wrong", or "1 minus the probability of the right answer"? Because log loss has three properties that make it ideal.

**It punishes confident mistakes hard.**

| Probability given to the correct token | Loss (−ln p) | Interpretation |
| --- | --- | --- |
| 0.99 | 0.01 | Confident and right: almost no penalty |
| 0.5 | 0.69 | Hedging |
| 0.1 | 2.30 | Mostly wrong |
| 0.01 | 4.61 | Confidently wrong |
| 0.0001 | 9.21 | Almost ruled the truth out |
| 0 | infinite | Said the truth was impossible |

A model can never say "impossible" about something that then happens without paying an infinite price. That keeps it humble. Softmax helps here too: because eˣ is never zero, every token always gets at least a tiny probability.

**It rewards honesty.** The loss is lowest, on average, when the model reports its true belief. If the real chance of "mat" is 40%, saying 40% beats saying 90% (overconfident) or 10% (underconfident). Statisticians call such a scoring rule **proper**: the model cannot game it.

**It gives useful signals for learning.** The loss changes smoothly as the probabilities change, so calculus can say exactly which way to nudge each parameter (Part IV). A plain "right/wrong" count would give no such guidance: nudging a parameter slightly usually changes nothing, so there would be no slope to follow.

### 9.6 Perplexity: loss you can picture

A loss of 1.89 is hard to interpret. **Perplexity** turns it into something intuitive:

$$
\text{perplexity} = e^{\text{average loss}}
$$

For our sentence, e¹·⁸⁹ ≈ 6.6. Read it as: on average, the model was as uncertain as if it were choosing uniformly among about 6.6 equally likely words.

| Model | Perplexity | Intuition |
| --- | --- | --- |
| Random guessing over 100,000 tokens | 100,000 | No idea at all |
| A simple counting model | Hundreds | Narrows it down a lot |
| A strong modern LLM on ordinary text | Single digits | Torn between a handful of options |
| A perfect oracle | 1 | Always certain and always right |

The first row hides a useful fact: an untrained model starts with loss around ln(100,000) ≈ 11.5, because it spreads probability evenly. Watching the loss fall from there is how researchers track training (Chapter 12).

### 9.7 KL divergence: the gap between beliefs

Cross-entropy contains two parts: the unavoidable uncertainty of the world (the entropy of P) and the extra surprise caused by having wrong beliefs. That extra part has its own name, the **Kullback–Leibler (KL) divergence**:

$$
D_{KL}(P \,\|\, Q) = H(P, Q) - H(P) = \sum_x P(x) \ln \frac{P(x)}{Q(x)}
$$

It is zero when Q matches P exactly and positive otherwise. You will meet it again in Chapter 16: when a model is fine-tuned with human feedback, a KL term keeps the new model from drifting too far from the original, like a leash.

> **Developer note.** In PyTorch the whole loss, softmax included, is one call:
>
> ```python
> loss = torch.nn.functional.cross_entropy(logits, targets)
> ```
>
> Here `logits` has shape (number of positions, vocabulary size) and `targets` holds the ID of the correct token at each position.

### Chapter summary

| Idea | Formula | Plain meaning |
| --- | --- | --- |
| Surprise | −log p | Rare events tell you more |
| Entropy | −Σ P log P | Average surprise; how uncertain a distribution is |
| Cross-entropy | −Σ P log Q | Average surprise when your beliefs Q differ from the truth P |
| LLM loss | −ln Q(correct token), averaged | How surprised the model was by the real text |
| Perplexity | e^loss | The number of equally likely options it felt torn between |
| KL divergence | H(P, Q) − H(P) | Extra surprise caused by wrong beliefs |

> **Try it yourself.**
>
> 1. A model gives the correct next token probability 0.25. What is its loss in nats? In bits?
> 2. What is the entropy, in bits, of a fair 8-sided die?
> 3. A model's average loss is 2.3. What is its perplexity? (Hint: e²·³ ≈ 10.)
