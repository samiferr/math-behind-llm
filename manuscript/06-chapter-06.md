## Chapter 6 — Tokens and Embeddings: Turning Words into Vectors

An LLM first cuts text into pieces called tokens, then swaps each token for a learned vector called its embedding. This is the bridge between language and math: after this step, everything is vectors and matrices. In this chapter you will see how the cutting works, why the embedding is just a row of a big matrix, and how meaning ends up encoded as direction.

### 6.1 How should we cut up text?

Before a model can turn text into numbers, it must decide on the units. There are three obvious choices, and each has a problem:

| Unit | Vocabulary size | Problem |
| --- | --- | --- |
| Characters (a, b, c, …) | \~100 to a few thousand | Sentences become very long; each character carries almost no meaning |
| Whole words | Millions (every name, typo, number, and form like "run/runs/running") | New words like "rizz" or a product name are unknown |
| **Subword pieces** | **50,000–200,000** | **The compromise LLMs use** |

Subword tokens work like a well-stocked Lego set. Common words get their own single brick ("the", "cat", "house"). Rare words are built from smaller bricks: "unbelievably" might become "un" + "believ" + "ably". Any text at all can be built, because in the worst case you can fall back to single characters or even raw bytes.

![Figure 6.1 · One plausible tokenization; real tokenizers differ](figures/fig-06-01.png)

### 6.2 How a tokenizer learns its pieces

The most common method is called **byte-pair encoding** (BPE). Its idea is beautifully simple: repeatedly glue together the pair of neighbouring pieces that appears most often.

Imagine a tiny training text made of these words, with how often each occurs: "low" (5 times), "lower" (2), "newest" (6), "widest" (3).

1. Start with single letters: l, o, w, e, r, n, s, t, i, d.
2. Count every adjacent pair. "e" followed by "s" appears 9 times (6 in newest, 3 in widest). So do "s t". Merge one of them: add the new piece **es**.
3. Count again. Now "es" + "t" appears 9 times. Add **est**.
4. "l" + "o" appears 7 times (low and lower). Add **lo**, then **low**.
5. Keep going until the vocabulary reaches the target size.

The result: common chunks like "est" and "low" become single tokens, which is efficient, while rare words are still spelled out from smaller pieces. It is similar to how people invent abbreviations for phrases they write often, like "FYI" or "ASAP".

### 6.3 Tokens are not words

A useful rule of thumb for English is that one token is about 4 characters, or about three quarters of a word. 100 tokens is roughly 75 words. Some consequences you may have noticed when using AI tools:

- **Pricing and limits are in tokens**, not words. A "200,000-token context" holds roughly 150,000 English words.
- **Other languages often cost more tokens.** Tokenizers trained mostly on English text have fewer ready-made pieces for, say, Arabic or Hindi, so the same sentence splits into more tokens.
- **Spelling and counting letters can trip models up.** The model sees "strawberry" as perhaps two or three tokens, not ten letters, so "how many r's are in strawberry?" is harder for it than you would expect.
- **Numbers get chopped oddly.** "1234567" might become "123", "456", "7". That is one reason arithmetic on long numbers is hard for models.

### 6.4 First attempt at numbers: one-hot vectors

Now each token has an ID. How do we turn the ID into a vector? The simplest idea is a **one-hot vector**: a vector as long as the vocabulary, all zeros except a single 1 at the token's position.

With a toy vocabulary of 5 words \[cat, dog, mat, sat, the\]:

| Token | One-hot vector |
| --- | --- |
| cat | \[1, 0, 0, 0, 0\] |
| dog | \[0, 1, 0, 0, 0\] |
| mat | \[0, 0, 1, 0, 0\] |

One-hot vectors have two serious flaws:

1. **They are huge.** With 100,000 tokens, each vector has 100,000 entries, of which 99,999 are zero.
2. **They say nothing about meaning.** Take the dot product of "cat" and "dog": 1×0 + 0×1 + 0 + 0 + 0 = 0. The same is true for "cat" and "democracy". Every pair of words looks equally unrelated, which throws away everything we know about language.

We want short vectors where similar words are similar. That is an embedding.

### 6.5 The embedding table

An **embedding** is a short, dense vector (every entry is a meaningful number, not mostly zeros) assigned to each token. All the embeddings are stored as the rows of one big matrix called the **embedding table** or **embedding matrix**, with one row per token in the vocabulary.

If the vocabulary has V tokens and each embedding has d numbers, the table is V × d. For a toy model with 5 words and d = 3:

| Token (row) | Dim 1 | Dim 2 | Dim 3 |
| --- | --- | --- | --- |
| cat | 0.8 | 0.6 | 0.1 |
| dog | 0.7 | 0.7 | 0.2 |
| mat | −0.1 | 0.2 | 0.9 |
| sat | 0.0 | −0.6 | 0.5 |
| the | 0.1 | 0.0 | 0.0 |

To embed a token, look up its row. That's it. "Cat" becomes \[0.8, 0.6, 0.1\].

Here is a lovely connection to Chapter 5: looking up a row is the same as multiplying the one-hot vector by the embedding matrix. The single 1 picks out exactly one row, and all the zeros wipe out the rest:

$$
[1, 0, 0, 0, 0] \cdot E = \text{row 1 of } E = [0.8, 0.6, 0.1]
$$

In practice computers just look up the row directly because it is faster, but mathematically the embedding step is a matrix multiplication like everything else.

Now check similarity with the dot product from Chapter 4. Cat · dog = 0.56 + 0.42 + 0.02 = 1.00. Cat · mat = −0.08 + 0.12 + 0.09 = 0.13. The embeddings say "cat" is much more like "dog" than like "mat", which is exactly what we wanted and what one-hot vectors could never say.

> **Real sizes.** A real model might have V = 128,000 tokens and d = 4,096. Its embedding table is 128,000 × 4,096, about 524 million numbers. That single table can be several percent of the whole model.

### 6.6 Where do the numbers come from?

Nobody writes the embedding table by hand. It starts as random numbers and is **learned** during training, along with everything else. Why would learning produce meaningful vectors?

The answer is an old idea from linguistics called the **distributional hypothesis**, famously summed up by the linguist J. R. Firth: "You shall know a word by the company it keeps."

Consider these sentences:

- I fed the \_\_\_ this morning.
- The \_\_\_ chased a ball across the yard.
- We took the \_\_\_ to the vet.

"Dog" fits all three. So do "puppy" and, mostly, "cat". "Carburetor" fits none. A model that must predict the blanks will do better if it gives dog, puppy and cat similar vectors, because then whatever it learns about one helps with the others. So training pushes words that appear in similar contexts toward similar vectors. Meaning emerges from prediction.

### 6.7 A map of meaning

Embeddings have thousands of dimensions, but we can squash them down to two for a picture, using techniques that try to keep neighbours as neighbours. The result looks like a map of a city where each neighbourhood holds one kind of word:

![Figure 6.2 · Schematic map; positions illustrate the pattern, not a specific model](figures/fig-06-02.png)

Animals cluster together. So do cities, foods and action verbs. Notice also that within a cluster, closer words are more alike: Toronto and Montreal, both Canadian, sit nearer each other than either does to Tokyo.

> **Common confusion.** The two axes of such a map do not have names like "animal-ness" or "size". They are whatever directions the squashing method found convenient. Even in the full embedding, individual dimensions rarely mean one clean thing; meaning is usually spread across many dimensions at once.

### 6.8 Directions carry meaning

The most famous discovery about word embeddings is that *directions*, not just positions, carry meaning. The arrow from "man" to "woman" points in roughly the same direction as the arrow from "king" to "queen", from "uncle" to "aunt", and from "actor" to "actress".

![Figure 6.3 · Schematic; real analogies hold only approximately](figures/fig-06-03.png)

This lets you do arithmetic with meaning:

$$
\vec{\text{king}} - \vec{\text{man}} + \vec{\text{woman}} \approx \vec{\text{queen}}
$$

Read it as: start at "king", remove the "male" part, add the "female" part, and the nearest word to where you land is "queen". Other directions behave similarly in good embeddings: country to capital (France → Paris, Japan → Tokyo), present to past tense (walk → walked, swim → swam), and singular to plural.

These patterns are approximate, not perfect. They are clearest in older, simpler word-embedding models and messier inside large modern LLMs. But the core lesson holds: **in an embedding space, a direction can stand for a concept.** Researchers studying LLMs today regularly find directions for things like sentiment, truthfulness or language, and can even nudge a model's behaviour by adding such a direction to its internal vectors.

> **A note on bias.** Embeddings learn from human writing, so they also learn human biases. Early studies found, for example, that "man is to computer programmer as woman is to homemaker" emerged from some embeddings. Measuring and reducing such biases is an active area of work.

### 6.9 The same word, different meanings

The embedding table gives each token exactly one vector. But many words have several meanings:

- I deposited money at the **bank**.
- We had a picnic on the river **bank**.

The table gives both "bank"s the same starting vector, a blend of all its senses. That is fine, because the table is only the starting point. The transformer layers that follow (Chapters 14 and 15) update each token's vector using the words around it. After a few layers, the "bank" next to "river" has drifted toward water and nature, and the "bank" next to "money" has drifted toward finance. These updated vectors are called **contextual embeddings**, and producing them is the whole job of the transformer.

### 6.10 One more ingredient: position

There is a subtle problem. The embedding of "dog" is the same wherever "dog" appears. So from embeddings alone, "dog bites man" and "man bites dog" contain exactly the same three vectors, just in a different order, and attention (as you will see) does not naturally care about order. The model needs to know which token came first. Chapter 15 shows how position information is mixed into the vectors. For now, just remember that word order must be added on purpose.

> **Developer note.** In PyTorch, the embedding table is one line:
>
> ```python
> emb = torch.nn.Embedding(num_embeddings=50000, embedding_dim=768)
> vectors = emb(torch.tensor([3118, 32923, 2735]))   # shape (3, 768)
> ```

### Chapter summary

- Text is split into subword **tokens** by a tokenizer such as BPE; each token has an ID.
- One-hot vectors are huge and treat all words as equally unrelated.
- An **embedding table** (V × d matrix) stores a short, learned vector per token; embedding is a row lookup.
- Training makes words used in similar contexts get similar vectors.
- Directions in embedding space can represent concepts (king − man + woman ≈ queen).
- Initial embeddings ignore context and position; the transformer layers add both.

> **Try it yourself.**
>
> 1. Using the toy table in 6.5, compute dog · sat and mat · sat. Which word is "sat" more similar to?
> 2. If a model has a vocabulary of 32,000 and d = 4,096, how many numbers are in its embedding table?
> 3. Paste a paragraph into an online tokenizer tool and count tokens versus words. Is the 4-characters-per-token rule close?
