# Part V — The Transformer: Putting It All Together

## Chapter 14 — Attention, Step by Step

Attention is how each word in a sentence gathers information from the other words that matter to it. Mathematically, it is just three ideas you already know: dot products to score relevance (Chapter 4), softmax to turn scores into weights (Chapter 8), and a weighted mixture of vectors (Chapter 3). This chapter puts them together and computes attention by hand.

### 14.1 Why words need context

Read these two sentences:

- I sat on the river **bank**.
- I deposited a cheque at the **bank**.

Same word, two meanings. You decided which by looking at other words: "river" in the first, "deposited" and "cheque" in the second. Now this one:

> The animal didn't cross the street because **it** was too tired.

What does "it" refer to? The animal, obviously, because streets don't get tired. Change one word, "because it was too *wide*", and "it" now means the street. To understand a word, you must look at the right other words. Which words are "right" depends on the sentence.

In Chapter 6, each token started with a fixed embedding that knows nothing about its neighbours. Attention is the mechanism that lets each token's vector absorb information from the relevant neighbours, turning "bank in general" into "bank as in riverside".

### 14.2 The core idea: a weighted average of the other words

The simplest possible way to mix in context would be to replace each word's vector with the average of all the word vectors in the sentence. That brings in context, but treats every word as equally relevant, which is wrong: for "it", the word "animal" matters far more than "the".

Attention does a **weighted** average instead, where the weights are computed on the fly for each word and each sentence:

$$
\text{new vector for word } i = \sum_j \alpha_{ij} \, \mathbf{v}_j
$$

Here αᵢⱼ (alpha) is how much word *i* attends to word *j*. For each word *i*, the weights are positive and add up to 1, so this is exactly the smoothie mixture from Chapter 3: "62% animal, 12% street, a little of everything else".

The whole question of attention is: **how do we compute good weights?**

### 14.3 Queries, keys and values

Think of searching for a video online:

1. You type a **query**: "easy pasta recipe".
2. Every video has a title and tags, its **key**, that describe what it contains.
3. The site compares your query to every key and ranks the matches.
4. You watch the matching videos and take away their **value**: the actual content.

Attention gives every token all three roles at once. From each token's vector **x**, the model computes three new vectors by multiplying by three learned matrices (Chapter 5):

| Vector | Formula | Question it answers | Video-search analogy |
| --- | --- | --- | --- |
| Query **q** | x W\_Q | What am I looking for? | Your search text |
| Key **k** | x W\_K | What do I offer, to anyone searching? | A video's title and tags |
| Value **v** | x W\_V | If someone attends to me, what information do I hand over? | The video itself |

![Figure 14.1 · Query, key and value in one attention step](figures/fig-14-01.png)

Why three separate vectors instead of using **x** for everything? Because "what I'm looking for" and "what I contain" are different. The word "it" is *looking for* a noun that could be its referent; that is a different thing from what "it" itself *contains*. Separate matrices let the model learn these roles independently. All three matrices are learned by gradient descent, like every other weight.

### 14.4 The attention formula, one step at a time

For one word *i*:

1. **Score** every word *j* (including itself) by the dot product of *i*'s query with *j*'s key: scoreᵢⱼ = qᵢ · kⱼ. A high score means "j has what i is looking for".
2. **Scale** each score by dividing by √d, where d is the length of the key vectors. (Chapter 4, section 4.8: this keeps scores from growing huge in long vectors, which would make softmax put all the weight on one word and stop learning.)
3. **Softmax** the scaled scores across all *j*, turning them into weights αᵢⱼ that are positive and sum to 1.
4. **Mix** the value vectors with those weights: outputᵢ = Σⱼ αᵢⱼ vⱼ.

Do it for every word at once with matrices, and you get the famous formula from the 2017 paper "Attention Is All You Need", which introduced the transformer:

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{T}}{\sqrt{d_k}}\right) V
$$

Read it inside out: QKᵀ computes every query-key dot product at once (a table of scores, one row per word); dividing by √dₖ scales them; softmax turns each row into weights; multiplying by V mixes the values. Every symbol in it is something you have already learned.

### 14.5 A worked example by hand

Take the phrase "the river bank" and compute the new vector for "bank". To keep the arithmetic small, vectors have just 2 numbers. Suppose the learned matrices have already produced:

| Token | Key k | Value v (think: \[nature, finance\]) |
| --- | --- | --- |
| the | \[0.1, 0.0\] | \[0, 0\] |
| river | \[1.0, 1.0\] | \[1.0, 0.0\] |
| bank | \[0.5, 0.5\] | \[0.5, 0.5\] |

The query for "bank" is q = \[2, 1\]. Now follow the four steps.

**Step 1 — Scores (q · k):**

- the: 2 × 0.1 + 1 × 0 = 0.2
- river: 2 × 1 + 1 × 1 = 3.0
- bank: 2 × 0.5 + 1 × 0.5 = 1.5

**Step 2 — Scale** by √2 ≈ 1.414: \[0.14, 2.12, 1.06\].

**Step 3 — Softmax:**

| Token | Scaled score | e^score | Weight |
| --- | --- | --- | --- |
| the | 0.14 | 1.15 | 0.09 |
| river | 2.12 | 8.34 | 0.67 |
| bank | 1.06 | 2.89 | 0.23 |
| **Sum** |  | **12.38** | **1.00** |

(Weights are rounded; unrounded, they sum to exactly 1.)

**Step 4 — Mix the values:**

$$
0.09 \times [0, 0] + 0.67 \times [1, 0] + 0.23 \times [0.5, 0.5] \approx [0.79, 0.12]
$$

The original "bank" value was \[0.5, 0.5\]: equally nature and finance, ambiguous. After attention, "bank" has become \[0.79, 0.12\]: strongly "nature". It looked at "river", found a strong match, and absorbed its meaning. That is attention.

> **Try it yourself.** Change the phrase to "the money bank" by giving "money" the key \[1.0, 1.0\] and value \[0.0, 1.0\]. Keep everything else the same. What does "bank" become? (The weights are unchanged, so the output is about \[0.12, 0.79\]: finance.)

### 14.6 No peeking: the causal mask

An LLM is trained to predict the next word. If, during training, the word "cat" could attend to the word that comes after it, it could simply copy the answer. That would be cheating, and the model would learn nothing useful.

So LLMs use a **causal mask**: each word may attend only to itself and the words before it. In practice, the scores for future positions are set to −∞ before softmax. Since e⁻∞ = 0, those positions get exactly zero weight.

![Figure 14.2 · Illustrative weights for one attention head](figures/fig-14-02.png)

The result is a triangle: "The" sees only itself, "cat" sees two words, and "mat" sees all six. This is why these models are called **decoder-only** or **autoregressive**: they read left to right, like writing a sentence one word at a time, never knowing what comes next. The mask also means training is efficient: one pass over a sentence of 1,000 tokens gives 1,000 next-token predictions at once, each using only the past.

### 14.7 What attention learns

Nobody tells the model which words should attend to which. The query and key matrices are learned, and useful patterns emerge. Visualizations of trained models show heads that link pronouns to the nouns they refer to, like this:

![Figure 14.3 · Illustrative weights in the style of published attention visualizations](figures/fig-14-03.png)

Other heads learn patterns such as:

- Attend to the previous word (useful for local grammar).
- Attend from a verb to its subject ("the keys to the cabinet **are**" must agree with "keys", not "cabinet").
- Attend from a closing bracket or quote back to the opening one.
- Copy patterns: if "Mr. Dursley" appeared earlier, then after "Mr." attend to what followed "Mr." last time. Researchers call these **induction heads**, and they are thought to be one basis of the in-context learning that lets LLMs follow examples in a prompt.

### 14.8 Multi-head attention: a committee of specialists

One attention pattern per layer would be limiting. "Bank" may need to look at "river" for meaning, at "the" for grammar, and at a word 50 tokens earlier for the topic, all at once.

So transformers run several attention operations in parallel, called **heads**, each with its own W\_Q, W\_K and W\_V matrices. Think of a hiring committee: one member focuses on technical skills, another on communication, another on culture fit. Each forms an opinion, and the opinions are combined.

The arithmetic is neat. With a hidden size of 4,096 and 32 heads, each head works with vectors of 4,096 / 32 = 128 numbers. The 32 outputs of 128 numbers are glued end to end (concatenated) back into 4,096 numbers, then mixed by one more matrix, W\_O:

$$
\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \dots, \text{head}_h) \, W_O
$$

The total cost is about the same as one big head, but the model gets 32 different ways of looking.

### 14.9 The price of attention

Attention compares every token with every other token. For n tokens, that is n × n scores:

| Context length (tokens) | Scores per head per layer |
| --- | --- |
| 1,000 | 1 million |
| 10,000 | 100 million |
| 100,000 | 10 billion |

Doubling the context quadruples this work. That is why long context windows were hard to achieve and why researchers have invented many tricks to make attention cheaper.

One trick matters for every chatbot: the **KV cache**. When generating text one token at a time, the keys and values of earlier tokens never change (thanks to the causal mask). So the model stores them and, for each new token, computes only the new token's query, key and value. Without this cache, generating each new word would mean recomputing the whole conversation from scratch.

> **Developer note.** Single-head attention in a few lines of NumPy:
>
> ```python
> def attention(Q, K, V, causal=True):
>     scores = Q @ K.T / np.sqrt(K.shape[-1])
>     if causal: scores += np.triu(np.full(scores.shape, -np.inf), k=1)
>     w = np.exp(scores - scores.max(axis=-1, keepdims=True))
>     w /= w.sum(axis=-1, keepdims=True)
>     return w @ V
> ```

### Chapter summary

- Attention lets each token build a new vector as a weighted mixture of other tokens' values.
- Each token produces a query (what it seeks), a key (what it offers) and a value (what it shares), using learned matrices.
- Weights = softmax of query-key dot products, scaled by √d.
- The causal mask blocks attention to future tokens, so the model cannot cheat.
- Multiple heads look for different relationships in parallel; their outputs are concatenated and mixed.
- Attention costs grow with the square of the context length; the KV cache speeds up generation.

> **Try it yourself.**
>
> 1. For a query \[1, 0\] and keys \[1, 0\], \[0, 1\], \[−1, 0\], compute the attention weights (use √d = √2 ≈ 1.41).
> 2. In a causal model, how many tokens can the 5th token attend to?
> 3. A model has hidden size 2,048 and 16 heads. What is each head's dimension?
