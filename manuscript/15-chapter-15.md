## Chapter 15 — The Transformer Block

A transformer block is attention followed by a feed-forward network, with two helpers wrapped around each: a **residual connection** that adds the result back onto the input, and **layer normalization** that keeps the numbers at a steady scale. An LLM is a stack of identical blocks, plus a way to tell the model the order of the words. This chapter assembles those pieces.

### 15.1 The block at a glance

![Figure 15.1 · A pre-norm transformer block, the layout used by most current LLMs](figures/fig-15-01.png)

Follow one token's vector through the block:

1. **Normalize** it (15.4).
2. **Attention:** gather information from other tokens (Chapter 14).
3. **Add** the attention result back onto the original vector (15.3).
4. **Normalize** again.
5. **Feed-forward:** process the vector on its own (Chapter 13).
6. **Add** that result back on as well.

The output has exactly the same shape as the input: one vector of d numbers per token. That is what makes it possible to stack blocks: the output of block 1 is a perfectly good input for block 2.

### 15.2 Telling the model about word order

Attention, as you saw in Chapter 14, is a weighted sum. Sums do not care about order: 2 + 5 + 3 is the same as 3 + 2 + 5. So without extra help, a transformer would see "dog bites man" and "man bites dog" as the same bag of words. We must inject position information.

**Idea 1: learned positions.** Give each position (1st, 2nd, 3rd, …) its own learned vector, just like token embeddings, and add it to the token's embedding. Simple, and used by early models like GPT-2. The downside: the model cannot handle positions beyond the longest it was trained on.

**Idea 2: waves.** The original transformer added a fixed pattern of sine and cosine waves of different speeds. Think of a clock: the second hand moves fast, the minute hand slower, the hour hand slower still. Together, the three hands identify any moment of the day uniquely. Positional waves do the same for positions in a sentence:

![Figure 15.2 · Three sine waves, computed; real encodings use hundreds of frequencies](figures/fig-15-02.png)

Pick any position on the chart and read the three values: that triple is unique to that position. Fast waves distinguish neighbours; slow waves distinguish distant positions.

**Idea 3: rotation (RoPE).** Most current LLMs use **rotary position embeddings**. Instead of adding anything to the vectors, they *rotate* each query and key by an angle proportional to its position. Picture two clock hands: one rotated for position 10, one for position 13. When you take their dot product (the attention score, Chapter 4), what matters is the angle *between* them, which depends only on the distance 3, not on whether the words are at 10 and 13 or at 510 and 513. So attention naturally becomes sensitive to **relative** position, "this word is three before that one", which is exactly what grammar cares about.

### 15.3 Residual connections: keep the original, add the changes

A residual connection is the simplest idea in this chapter:

$$
\mathbf{x}_{\text{new}} = \mathbf{x} + \text{Sublayer}(\mathbf{x})
$$

The sub-layer (attention or feed-forward) does not replace the vector. It computes a *change*, and the change is added on. This is like editing a shared document with suggestions: the original text stays, and each reviewer layers their edits on top.

Researchers often picture the stack of residual connections as a **residual stream**: a running vector for each token that flows from the bottom of the model to the top. Every attention layer and every feed-forward layer reads from the stream and writes an update back into it. It is a shared whiteboard that every layer can read and annotate.

Residual connections have two big benefits:

- **Easy defaults.** If a layer has nothing useful to add, it can learn to output almost zero, and the vector passes through untouched. Adding layers can never make things much worse.
- **Gradients flow.** In Chapter 11 you saw that gradients can vanish when multiplied through many layers. The "+ x" creates a direct path: the derivative of x with respect to itself is 1, so the gradient can travel from the top of the model to the bottom without shrinking away. This is what makes 80-layer models trainable.

### 15.4 Layer normalization: keep the numbers in range

As vectors pass through many layers and get many updates added on, their numbers can drift: some getting very large, others tiny. Large drifts make training unstable. **Layer normalization** rescales each token's vector so that its numbers have an average of 0 and a spread (standard deviation) of 1.

An everyday analogy: a teacher who "grades on a curve" converts raw scores into "how far above or below average, in typical-spread units". A score of 85 on an easy test and 70 on a hard test might both become "+1", meaning "one spread above average". Normalization keeps only the shape of the numbers, not their overall size.

The recipe for a vector x with d numbers:

1. Compute the mean: μ = (sum of the numbers) / d.
2. Compute the standard deviation: σ = √(average of (number − μ)²).
3. Replace each number by (number − μ) / σ.
4. Multiply by a learned scale γ and add a learned shift β, so the model can undo the normalization where useful.

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sigma} + \beta
$$

**Worked example.** x = \[2, 4, 6\].

- Mean: (2 + 4 + 6) / 3 = 4.
- Differences from the mean: \[−2, 0, 2\]. Squared: \[4, 0, 4\]. Average: 8/3 ≈ 2.67. Standard deviation: √2.67 ≈ 1.63.
- Normalized: \[−2 / 1.63, 0, 2 / 1.63\] ≈ \[−1.22, 0, 1.22\].

The vector \[200, 400, 600\] would normalize to exactly the same thing: only the relative pattern survives. Many recent LLMs use a simpler variant called **RMSNorm**, which skips subtracting the mean and only divides by the root-mean-square size. The purpose is the same: steady numbers, stable training.

### 15.5 Stacking blocks: an assembly line of understanding

A model stacks N identical blocks (identical in design, each with its own learned weights). As vectors rise through the stack, they get progressively richer. Studies of trained models suggest a rough division of labour:

| Depth | What tends to be represented | Analogy |
| --- | --- | --- |
| Early layers | Word pieces, spelling, local grammar | Recognizing letters and words |
| Middle layers | Meaning in context, who-did-what-to-whom, facts | Understanding sentences |
| Late layers | What to say next: style, the specific next token | Choosing your next words |

This is a tendency, not a strict rule; information is spread throughout. But it captures why depth helps: each layer builds on what the previous ones prepared, like stations on an assembly line.

### 15.6 The top of the model: back to words

After the last block, each token's vector is normalized one final time. Then, for the **last** token only (when generating), the model asks which vocabulary word should come next:

1. Multiply the final vector by an **unembedding matrix** of shape d × V. This takes the dot product of the vector with one learned vector per vocabulary token (Chapter 4), producing V logits.
2. Apply softmax (Chapter 8) to get a probability for every token.

Many models reuse the embedding table itself as the unembedding matrix, a trick called **weight tying**: the same vector that says "what 'mat' means" when reading is used to score "how much does this context call for 'mat'" when writing.

### 15.7 The whole model in one picture

Putting Chapters 6 to 15 together, here is a complete decoder-only LLM:

$$
\begin{aligned}
X_0 &= \text{Embed}(\text{tokens}) \\
X_{\ell} &= X_{\ell-1} + \text{Attention}(\text{Norm}(X_{\ell-1})) \\
X_{\ell} &\leftarrow X_{\ell} + \text{FFN}(\text{Norm}(X_{\ell})) \qquad \text{for } \ell = 1, \dots, N \\
\text{probabilities} &= \text{softmax}(\text{Norm}(X_N) \, W_{\text{unembed}})
\end{aligned}
$$

(With RoPE, position is applied inside each attention layer rather than added at the start.)

And here are the published shapes of some well-known open models:

| Model | Layers (N) | Hidden size (d) | Attention heads | Parameters |
| --- | --- | --- | --- | --- |
| GPT-2 small (2019) | 12 | 768 | 12 | about 124 million |
| GPT-2 XL (2019) | 48 | 1,600 | 25 | about 1.5 billion |
| Llama 2 7B (2023) | 32 | 4,096 | 32 | about 7 billion |
| Llama 2 70B (2023) | 80 | 8,192 | 64 | about 70 billion |

Look at how little changes between rows: more layers, wider vectors, more heads. The design is the same. This is the deep lesson of modern AI: a simple, uniform architecture, scaled up with more data and computing power, keeps getting more capable.

> **Developer note.** A pre-norm transformer block in PyTorch:
>
> ```python
> class Block(nn.Module):
>     def __init__(self, d, heads):
>         super().__init__()
>         self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
>         self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
>         self.ffn = nn.Sequential(nn.Linear(d, 4*d), nn.GELU(), nn.Linear(4*d, d))
>     def forward(self, x, mask):
>         h = self.n1(x); x = x + self.attn(h, h, h, attn_mask=mask)[0]
>         return x + self.ffn(self.n2(x))
> ```

### Chapter summary

- A transformer block = norm → attention → add, then norm → feed-forward → add.
- Position must be injected: learned vectors, sine waves, or rotations (RoPE) that make attention sensitive to relative distance.
- Residual connections add each sub-layer's change onto a running stream, giving gradients a highway.
- Layer normalization (or RMSNorm) rescales each vector to a steady size.
- Blocks are stacked N times; a final norm, unembedding matrix and softmax turn the last vector into next-token probabilities.

> **Try it yourself.**
>
> 1. Layer-normalize \[1, 1, 4\] (without γ and β).
> 2. Why can't the model simply skip positional information, given that the words are fed in order?
> 3. A model has 24 blocks. How many attention sub-layers and how many feed-forward sub-layers does it have?
