## Chapter 4 — The Dot Product: Measuring Similarity

The dot product multiplies two vectors position by position and adds up the results, giving one number that says how strongly the two vectors agree. It is the single most-used calculation inside an LLM: every attention score and every next-word score is a dot product.

### 4.1 You already know the dot product: a till receipt

You buy 3 coffees at $2, 2 muffins at $3 and 1 juice at $4. What do you pay?

$$
3 \times 2 + 2 \times 3 + 1 \times 4 = 6 + 6 + 4 = 16
$$

Write the quantities as one vector and the prices as another:

- Quantities **q** = \[3, 2, 1\]
- Prices **p** = \[2, 3, 4\]

Your bill is: multiply matching positions, then add. That is the **dot product**, written with a dot between the vectors:

$$
\mathbf{q} \cdot \mathbf{p} = q_1 p_1 + q_2 p_2 + q_3 p_3 = \sum_{i=1}^{n} q_i p_i
$$

Notice two things:

1. The two vectors must have the same length. You cannot price four items with three prices.
2. The result is a **single number**, not a vector. Two lists go in; one number comes out.

You also met the dot product in Chapter 2 under another name: the weighted sum. The house price was a dot product between the house's features \[sqft, bedrooms, age\] and the weights \[250, 15,000, −1,000\], plus a base.

> **Try it yourself.** Compute \[1, 2, 3\] · \[4, 5, 6\]. (Answer: 4 + 10 + 18 = 32.)

### 4.2 What the number means: agreement

The receipt example shows *how* to compute a dot product. A movie example shows *what it means*.

A streaming service describes each movie with three scores from −1 to 1: \[action, romance, comedy\]. It describes your taste the same way: how much you like each genre.

- Your taste: **u** = \[0.9, −0.5, 0.3\] (love action, dislike romance, mildly like comedy)
- Movie A, a heist thriller: \[0.8, −0.2, 0.4\]
- Movie B, a period romance: \[−0.6, 0.9, 0.1\]

|  | Action | Romance | Comedy | Dot product with your taste |
| --- | --- | --- | --- | --- |
| Movie A | 0.9 × 0.8 = 0.72 | −0.5 × −0.2 = 0.10 | 0.3 × 0.4 = 0.12 | **0.94** |
| Movie B | 0.9 × −0.6 = −0.54 | −0.5 × 0.9 = −0.45 | 0.3 × 0.1 = 0.03 | **−0.96** |

Movie A scores high; recommend it. Movie B scores negative; hide it.

Look at why. Each product in the row is positive when your taste and the movie agree (both positive, or both negative), and negative when they disagree. Adding them up tallies the total agreement. So:

- **Large positive dot product:** the vectors agree. They point in similar directions.
- **Near zero:** they are unrelated.
- **Large negative:** they disagree. They point in opposite directions.

This "agreement score" interpretation is the one to keep in your head for the rest of the book.

### 4.3 The geometry: angles

In the arrow picture, "pointing in similar directions" means a small angle between the arrows. There is an exact link between the dot product and that angle:

$$
\mathbf{a} \cdot \mathbf{b} = \|\mathbf{a}\| \, \|\mathbf{b}\| \cos(\theta)
$$

Here θ (the Greek letter theta) is the angle between the two arrows, and cos is the **cosine**, a function from trigonometry. You do not need trigonometry for this book. You only need to know what the cosine does:

![Figure 4.1 · Three angles, three cosine values](figures/fig-04-01.png)

| Angle between arrows | Cosine | Meaning |
| --- | --- | --- |
| 0° (same direction) | 1 | Perfect agreement |
| 60° | 0.5 | Partly similar |
| 90° (right angle) | 0 | Unrelated |
| 120° | −0.5 | Partly opposed |
| 180° (opposite) | −1 | Perfect disagreement |

The formula says the dot product mixes two things: **how long** the arrows are and **how aligned** they are. Long, aligned arrows give a big dot product. Short or misaligned ones give a small one.

### 4.4 Cosine similarity: alignment only

Sometimes length is a nuisance. Recall the book-review problem from Chapter 3: a long review has bigger word counts than a short one, so its vector is longer, and its dot products are bigger just because it rambles. To compare only direction, divide the dot product by both lengths:

$$
\text{cosine similarity}(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

The answer always lies between −1 and 1, whatever the vectors' lengths.

**Worked example.** Compare **a** = \[3, 4\] and **b** = \[6, 8\]. The second is just the first doubled, so they point the same way.

1. Dot product: 3 × 6 + 4 × 8 = 18 + 32 = 50
2. Lengths: ‖**a**‖ = √(9 + 16) = 5 and ‖**b**‖ = √(36 + 64) = 10
3. Cosine similarity: 50 / (5 × 10) = 1

Perfect similarity, as expected: doubling a vector changes its length, not its direction.

Now compare **a** = \[3, 4\] with **c** = \[4, −3\]:

1. Dot product: 12 − 12 = 0
2. Cosine similarity: 0. These arrows are at exactly 90°. Unrelated.

> **Common confusion.** Cosine similarity and distance (Chapter 3) are related but different. Two vectors can be far apart (one very long, one short) yet have cosine similarity 1 because they point the same way. Which one you use depends on whether length carries meaning in your problem. For word embeddings, direction usually matters most, so cosine similarity is the default.

### 4.5 Real-world use: semantic search

Old search engines matched keywords. Search "cheap flights to Paris" and they looked for pages containing those exact words. A page titled "Budget airfare to the French capital" might be missed.

Modern search, including the retrieval step behind many AI assistants, works with vectors:

1. Every document is turned into a vector by an embedding model, once, and stored.
2. Your query is turned into a vector the same way.
3. The system computes the cosine similarity between your query vector and every document vector.
4. It returns the documents with the highest scores.

Because similar meanings give similar directions, "cheap flights to Paris" and "budget airfare to the French capital" end up with a high cosine similarity even though they share no words. This approach is called **semantic search**, and when it feeds documents into an LLM's prompt it is called **retrieval-augmented generation (RAG)**.

> **Developer note.** In a Django project with PostgreSQL, the pgvector extension stores embeddings in a column and finds nearest neighbours in SQL. The operator `<=>` computes cosine distance, which is 1 minus cosine similarity:
>
> ```sql
> SELECT title FROM docs ORDER BY embedding <=> :query_vec LIMIT 5;
> ```

### 4.6 Similarity between words

Here is what cosine similarity looks like between word vectors from a trained model. The exact numbers vary by model; the pattern does not.

![Figure 4.2 · Illustrative values; real models give different numbers](figures/fig-04-02.png)

Words that appear in similar sentences ("I walked my \_\_\_ in the park") score highest. "Leash" and "veterinarian" are related by topic and score in the middle. "Car" and "democracy" are almost unrelated, with similarity near zero.

### 4.7 The dot product inside an LLM

The dot product appears in at least three central places in a transformer. You will study each in detail later; for now, just notice that one simple operation does all three jobs.

| Where | What is compared | What the result means | Chapter |
| --- | --- | --- | --- |
| Attention scores | A word's "question" vector with another word's "label" vector | How relevant is that other word to me? | 14 |
| Neurons in feed-forward layers | The token's vector with a neuron's weight vector | How strongly does this token match the pattern this neuron detects? | 13 |
| Final output scores | The final hidden vector with each vocabulary word's vector | How well does each word fit as the next token? | 17 |

The last row closes the loop with Chapter 1. Remember the bar chart of next-word probabilities for "The cat sat on the"? Each bar started life as a dot product. The model computed one vector summarizing "what should come next", took its dot product with the vector of every word in the vocabulary ("mat", "floor", "democracy", and 100,000 others), and the words whose vectors best agreed got the highest scores. Softmax then turned the scores into probabilities.

A model with a vocabulary of 100,000 tokens computes 100,000 dot products just to choose one next word. Computers handle this easily because, as Chapter 5 shows, all of those dot products can be done at once as a single matrix multiplication.

### 4.8 Why a big dot product can be a problem

One last detail you will need in Chapter 14. If vectors have thousands of components, their dot products can become very large simply because there are so many terms being added. Imagine scoring a movie on 4,096 genres instead of 3: even modest agreement on each adds up to a huge total.

Transformers fix this by dividing attention dot products by the square root of the vector length, √d. For d = 64, that means dividing by 8. This keeps the scores in a sensible range, and you will see exactly why it matters when we meet softmax.

### Chapter summary

- The dot product multiplies matching positions and adds them up: two vectors in, one number out.
- It measures agreement: positive when vectors point alike, zero when unrelated, negative when opposed.
- Cosine similarity divides by both lengths so only direction counts; it ranges from −1 to 1.
- Semantic search, recommendations and RAG all rank items by cosine similarity.
- Inside an LLM, dot products score attention, drive neurons and score every candidate next word.

> **Try it yourself.**
>
> 1. Compute the dot product of \[2, −1, 3\] and \[1, 4, 2\].
> 2. Find the cosine similarity of \[1, 0\] and \[1, 1\]. What angle is that? (Hint: cos 45° ≈ 0.707.)
> 3. Your taste is \[1, 1, −1\] for \[sci-fi, comedy, horror\]. Which movie should you watch: \[0.9, 0.2, 0.8\] or \[0.5, 0.7, −0.3\]?
