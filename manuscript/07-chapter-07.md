# Part III — Probability: How a Model Expresses Uncertainty

## Chapter 7 — Probability Basics: Putting Numbers on Uncertainty

Probability is a number between 0 and 1 that says how likely something is, and a language model is, at heart, a machine for estimating one particular probability: the chance of each possible next word given the words so far. This chapter builds that idea from coin flips up to the exact formula an LLM uses to score a whole sentence.

### 7.1 A number for "how likely"

Weather forecasts are the most familiar use of probability. "70% chance of rain" means: on days like this one, it rains about 7 times out of 10.

Probabilities are written as numbers from 0 to 1, or as percentages from 0% to 100%:

| Probability | Percentage | Meaning | Example |
| --- | --- | --- | --- |
| 0 | 0% | Impossible | Rolling a 7 on a normal die |
| 0.01 | 1% | Very unlikely | Being dealt a pair of aces in poker (about 0.5%) |
| 0.5 | 50% | Even odds | A fair coin landing heads |
| 0.9 | 90% | Very likely | A Toronto day in January being below 5°C |
| 1 | 100% | Certain | The sun rising tomorrow |

We write the probability of an event A as P(A). So P(heads) = 0.5.

### 7.2 Probability as counting

The simplest way to find a probability is to count:

$$
P(A) = \frac{\text{number of ways A can happen}}{\text{total number of equally likely outcomes}}
$$

A die has 6 faces. Two of them (5 and 6) are "greater than 4". So P(greater than 4) = 2/6 = 1/3 ≈ 0.33.

When outcomes are not equally likely, we count how often they actually happened instead. A coffee shop records that out of 1,000 orders last month, 420 were lattes. Its best estimate is P(latte) = 420 / 1,000 = 0.42. This is called a **frequency** estimate, and it is exactly how the simplest language models were built: count how often each word follows each other word in a huge pile of text.

### 7.3 More data, better estimates

If you flip a fair coin 10 times, you might easily get 7 heads. Does that mean P(heads) = 0.7? No. Small samples are noisy. But as you flip more and more, the share of heads settles toward the true value:

![Figure 7.1 · One simulated run of coin flips, values illustrative](figures/fig-07-01.png)

This is called the **law of large numbers**. It is why LLMs are trained on so much text: with trillions of words, the patterns the model learns are reliable rather than flukes.

### 7.4 Three rules of probability

You can combine probabilities with three simple rules.

**Rule 1 — Not.** The probability something does *not* happen is 1 minus the probability it does.

$$
P(\text{not } A) = 1 - P(A)
$$

If P(rain) = 0.7, then P(no rain) = 0.3.

**Rule 2 — Or (for things that cannot both happen).** Add them.

$$
P(A \text{ or } B) = P(A) + P(B)
$$

The probability of rolling a 1 or a 2 is 1/6 + 1/6 = 1/3. This rule has a crucial consequence: if a list of outcomes covers every possibility and no two can happen together, their probabilities **add up to exactly 1**. Every next-word distribution an LLM produces must obey this: the probabilities of all 100,000 tokens sum to 1.

**Rule 3 — And (for independent things).** Multiply them.

$$
P(A \text{ and } B) = P(A) \times P(B)
$$

Two coins both landing heads: 0.5 × 0.5 = 0.25. "Independent" means one does not affect the other. The coins do not talk to each other.

> **Common confusion.** Multiplying probabilities always makes them smaller (or equal), because you are multiplying numbers below 1. The chance of 10 heads in a row is 0.5¹⁰ ≈ 0.001. Long sequences of events, like long sentences, have tiny probabilities. This is normal and expected.

### 7.5 Conditional probability: "given that"

Most interesting events are not independent. The chance of carrying an umbrella depends on whether it is raining. We write the probability of A **given** B as P(A | B). The vertical bar is read "given".

$$
P(A \mid B) = \frac{P(A \text{ and } B)}{P(B)}
$$

In words: among only the situations where B happens, what fraction also have A? Conditioning means **zooming in** on a smaller set of situations.

**Worked example: a spam filter.** An email provider looks at 1,000 emails:

|  | Contains "FREE" | No "FREE" | Total |
| --- | --- | --- | --- |
| Spam | 240 | 60 | 300 |
| Not spam | 35 | 665 | 700 |
| Total | 275 | 725 | 1,000 |

- P(spam) = 300 / 1,000 = 0.30. Without other information, 30% of emails are spam.
- P(spam | contains "FREE") = 240 / 275 ≈ 0.87. Zooming in on emails containing "FREE", 87% are spam.

The word "FREE" changed the probability from 30% to 87%. That jump is what makes conditional probability powerful: evidence changes beliefs.

### 7.6 Language is conditional probability

Here is the key idea of this whole part of the book. What is the probability that the next word is "jelly"? On its own, tiny: "jelly" is a rare word. But given the context "peanut butter and", it becomes very likely:

![Figure 7.2 · Illustrative probabilities](figures/fig-07-02.png)

An LLM's entire job is to compute this:

$$
P(\text{next token} \mid \text{all previous tokens})
$$

for every possible next token, as accurately as possible. Everything else, attention, embeddings, billions of parameters, is machinery for estimating this one conditional probability well.

> **Real-world example.** Your phone's predictive keyboard, a search engine's autocomplete, and a doctor's diagnosis ("given these symptoms, how likely is flu?") all compute conditional probabilities. The LLM's version is just bigger: its "given" can be an entire book.

### 7.7 Building a language model by counting

Before neural networks, language models were built by counting. Let's build one from a tiny training text:

> the cat sat on the mat. the cat ate. the dog sat on the rug.

Count what follows each word. The word "the" appears 5 times followed by another word:

| Word after "the" | Count | Probability |
| --- | --- | --- |
| cat | 2 | 2 / 5 = 0.4 |
| mat | 1 | 1 / 5 = 0.2 |
| dog | 1 | 1 / 5 = 0.2 |
| rug | 1 | 1 / 5 = 0.2 |

So P(cat | the) = 0.4. Similarly, "sat" is always followed by "on", so P(on | sat) = 1. This is a **bigram model**: it looks only at the previous single word. A model that looks at the previous two words is a trigram model, and so on.

To generate text, start with "the", pick the next word using these probabilities, then repeat. You might get "the dog sat on the mat." Notice that sentence never appeared in the training text. Even this tiny model can combine pieces in new ways.

> **Developer note.** A bigram model is a dictionary of counters:
>
> ```python
> from collections import Counter, defaultdict
> counts = defaultdict(Counter)
> words = text.split()
> for a, b in zip(words, words[1:]): counts[a][b] += 1
> total = sum(counts['the'].values()); p_cat = counts['the']['cat'] / total
> ```

### 7.8 The chain rule: the probability of a whole sentence

How likely is the whole sentence "the cat sat"? Break it into one next-word decision at a time, and multiply:

$$
P(w_1, w_2, w_3) = P(w_1) \times P(w_2 \mid w_1) \times P(w_3 \mid w_1, w_2)
$$

![Figure 7.3 · The chain rule of probability, illustrative numbers](figures/fig-07-03.png)

This is the **chain rule of probability**, and it is exact, not an approximation. It holds for any sentence of any length:

$$
P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \dots, w_{t-1})
$$

The big Π (capital Greek pi) means "multiply all of these", just as Σ means "add all of these".

This formula explains why an LLM writes one token at a time. Generating a sentence is literally walking through this chain: compute the distribution for word 1, choose one, compute the distribution for word 2 given word 1, choose one, and so on.

It also explains why logarithms matter (Chapter 2). A 500-word answer has a probability that is the product of 500 numbers, each well below 1. The result might be around 10⁻⁴⁰⁰, smaller than a computer can store. Taking logs turns the product into a sum of manageable negative numbers:

$$
\log P(w_1, \dots, w_n) = \sum_{t=1}^{n} \log P(w_t \mid w_1, \dots, w_{t-1})
$$

This sum of log-probabilities is precisely what models are trained to make as large as possible (Chapter 9).

### 7.9 Why counting is not enough

If counting works, why do we need neural networks? Because language is too creative for counting. Consider:

> The quantum physicist's golden retriever refused to \_\_\_

This exact phrase has almost certainly never been written before, so its count is zero. A counting model has nothing to say. But you know the next word is probably something a dog might refuse to do: "eat", "walk", "come". You **generalize**: you know retrievers are dogs and dogs refuse walks, even though you have never seen this sentence.

This is the **sparsity** problem. With a vocabulary of 100,000 tokens, there are 100,000³ = 10¹⁵ possible three-word sequences; most will never appear even in trillions of words of text.

The solution is to replace counting with a **function** that turns any context into a probability distribution, and that treats similar contexts similarly. That is what embeddings provide (similar words get similar vectors, so "retriever" behaves like "dog"), and what a neural network does with them. Part IV shows how such a function learns; Part V shows its architecture.

### 7.10 A brief word on Bayes' rule

One famous formula connects P(A | B) and P(B | A):

$$
P(A \mid B) = \frac{P(B \mid A) \, P(A)}{P(B)}
$$

Check it with the spam table: P("FREE" | spam) = 240/300 = 0.8, P(spam) = 0.3, P("FREE") = 275/1,000 = 0.275. Then 0.8 × 0.3 / 0.275 ≈ 0.87, the same answer as before. Bayes' rule is central to statistics and older spam filters. LLMs do not use it directly, but you will see the phrase "Bayesian" in AI writing, and now you know what it refers to: updating beliefs when evidence arrives.

### Chapter summary

- A probability is a number from 0 to 1; P(A) is the probability of A.
- Probabilities can be estimated by counting; more data gives better estimates.
- "Not" subtracts from 1; "or" adds; "and" (independent) multiplies.
- P(A | B) is the probability of A given B: zoom in on cases where B holds.
- An LLM estimates P(next token | previous tokens).
- The chain rule multiplies next-token probabilities to score a whole sentence; logs turn this product into a sum.
- Counting fails for unseen contexts, which is why we need learned functions that generalize.

> **Try it yourself.**
>
> 1. A bag has 3 red and 7 blue marbles. What is P(red)? P(not red)?
> 2. From the toy text in 7.7, what is P(sat | cat)? P(ate | cat)?
> 3. Compute P("the cat ate") with the bigram model, assuming P(the) = 1 as the first word.
