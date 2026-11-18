# Part II — Linear Algebra: How Meaning Becomes Numbers

## Chapter 3 — Vectors: Lists of Numbers That Describe Things

A vector is simply an ordered list of numbers that describes something. A language model turns every word into a vector, so this humble list is the most important object in the whole book. In this chapter you will learn to add vectors, stretch them, measure their length and measure the distance between two of them.

### 3.1 Describing things with lists

You already use vectors every day without calling them that.

- **A weather report:** \[temperature 18°C, humidity 65%, wind 12 km/h, rain chance 30%\]
- **A hockey player's season:** \[goals 32, assists 41, games 82, penalty minutes 18\]
- **A paint colour on a screen:** \[red 255, green 128, blue 0\], which is orange
- **A bank client in a compliance system:** \[age 42, accounts 3, monthly transfers 12, average transfer $1,800\]

Each of these is a vector. Two rules make a list a vector:

1. **Order matters.** \[255, 128, 0\] is orange; \[0, 128, 255\] is sky blue. Same numbers, different positions, different meaning.
2. **Each position has a fixed meaning.** Position 1 is always red, position 2 always green. You must never mix them up between vectors.

We usually write vectors in square brackets and name them with a bold or arrowed letter:

$$
\mathbf{v} = [3, 1] \qquad \mathbf{c} = [255, 128, 0]
$$

Each number in the list is called a **component** or **element**. The number of components is the vector's **dimension**. The colour vector **c** has dimension 3.

### 3.2 Vectors as arrows

A two-number vector can be drawn as an arrow. Imagine giving directions in a city laid out in a grid: "walk 3 blocks east and 1 block north". That instruction is the vector \[3, 1\]. Draw an arrow from where you start to where you end and you have a picture of it.

The arrow has two properties:

- A **direction**: which way it points (mostly east, a bit north).
- A **length**: how far it goes.

This "arrow" view is the key intuition for LLMs. In a language model, the direction of a word's vector carries most of its meaning, and words with similar meanings point in similar directions. Hold on to that idea; Chapter 4 makes it precise.

### 3.3 Adding vectors

Suppose you walk \[3, 1\] and then walk \[1, 3\]. Where do you end up? Add the matching components:

$$
[3, 1] + [1, 3] = [3 + 1,\; 1 + 3] = [4, 4]
$$

![Figure 3.1 · Two walking directions added tip to tail](figures/fig-03-01.png)

Picture it as placing the second arrow at the tip of the first, then drawing the shortcut from start to finish. That shortcut is the sum.

Addition works the same in any number of dimensions. A non-arrow example: two grocery baskets, each described as \[apples, bread, milk\]:

|  | Apples | Bread | Milk |
| --- | --- | --- | --- |
| Your basket | 4 | 1 | 2 |
| Partner's basket | 2 | 0 | 1 |
| **Combined** | **6** | **1** | **3** |

You can only add vectors of the same dimension, and you always add position by position. Adding \[apples, bread\] to \[apples, bread, milk\] makes no sense, just as adding a 2-item receipt to a 3-item receipt column by column would misalign everything.

**Subtraction** works the same way and answers "what is the difference?". If last month's basket was \[4, 1, 2\] and this month's is \[6, 1, 3\], the change is \[6 − 4, 1 − 1, 3 − 2\] = \[2, 0, 1\]: two more apples and one more milk.

> **LLM connection.** Vector addition is used constantly inside a transformer. When a word's vector passes through a layer, the layer's result is *added* to the original vector rather than replacing it. This is called a **residual connection** (Chapter 15). It works like editing a document with tracked changes: the original is kept and adjustments are layered on top.

### 3.4 Scaling vectors

A recipe for 4 people needs \[flour 400 g, eggs 2, milk 300 ml\]. For 8 people you double every ingredient:

$$
2 \times [400, 2, 300] = [800, 4, 600]
$$

Multiplying a vector by a single number is called **scaling**, and the single number is called a **scalar** (because it scales). Every component gets multiplied by the same number.

In arrow terms:

- Multiplying by 2 makes the arrow twice as long, same direction.
- Multiplying by 0.5 makes it half as long.
- Multiplying by −1 flips it to point the opposite way.

> **Try it yourself.** Scale \[3, 1\] by 2, by 0.5 and by −1. Sketch the three arrows on graph paper. Which one points the opposite way?

### 3.5 Combining: weighted mixtures

Scaling and adding together let you make **mixtures**. A smoothie shop has two base blends, each a vector of \[sweetness, sourness, thickness\]:

- Berry blend: \[8, 3, 5\]
- Green blend: \[2, 4, 7\]

A "70% berry, 30% green" smoothie is:

$$
0.7 \times [8, 3, 5] + 0.3 \times [2, 4, 7] = [6.2, 3.3, 5.6]
$$

This is called a **weighted combination**, and it is the core operation of attention (Chapter 14). When an LLM reads "the bank of the river", the new vector for "bank" is literally a weighted mixture of the vectors of the surrounding words, with "river" getting a large weight. The weights are like the 70% and 30% in the smoothie.

### 3.6 Length: how long is the arrow?

Walk 3 blocks east and 4 blocks north. How far are you from where you started, as the crow flies? The ancient Pythagorean theorem answers: square each side, add, take the square root.

$$
\text{length} = \sqrt{3^2 + 4^2} = \sqrt{9 + 16} = \sqrt{25} = 5
$$

The length of a vector is also called its **magnitude** or **norm** and written with double bars. The rule extends to any number of dimensions: square every component, add them all up, take the square root.

$$
\|\mathbf{v}\| = \sqrt{v_1^2 + v_2^2 + \dots + v_n^2}
$$

For example, the length of \[2, 3, 6\] is √(4 + 9 + 36) = √49 = 7.

You cannot picture a 4,096-dimensional arrow. Nobody can. But you can compute its length with exactly the same recipe. This is the great trick of linear algebra: build intuition in 2 or 3 dimensions, where you can draw, then trust the arithmetic in thousands of dimensions, where you cannot.

### 3.7 Distance: how different are two things?

If vectors describe things, then the distance between two vectors measures how different those things are. To find it, subtract one vector from the other and take the length of the result.

$$
\text{distance}(\mathbf{a}, \mathbf{b}) = \|\mathbf{a} - \mathbf{b}\|
$$

Music streaming services use exactly this idea. Suppose each song is scored on two features, energy and acousticness, each from 0 to 1. Every song becomes a point:

![Figure 3.2 · Eight made-up songs, scores invented for illustration](figures/fig-03-02.png)

Two groups appear without anyone drawing them: loud, electronic songs at the bottom right, and quiet, acoustic songs at the top left. If you love "Rainy Sunday", the service can recommend its nearest neighbour, "Lullaby". The distance is:

$$
\sqrt{(0.18 - 0.08)^2 + (0.86 - 0.96)^2} = \sqrt{0.01 + 0.01} \approx 0.14
$$

Compare that with "Workout Anthem": √((0.18 − 0.92)² + (0.86 − 0.08)²) ≈ 1.08. Much farther away, so a much worse recommendation.

Real services use dozens or hundreds of features: tempo, mood, vocals, language, decade. The songs then live in a space you cannot draw, but distance is computed the same way.

> **LLM connection.** Words work like these songs. An LLM gives each word a vector of thousands of numbers. Words used in similar ways ("cat" and "kitten", or "Toronto" and "Montreal") end up near each other. Nobody tells the model which words are similar; it discovers this while learning to predict the next word, because similar words appear in similar sentences.

### 3.8 The average of several vectors

To find the average of several vectors, add them up and divide by how many there are, position by position. A hockey team's "average player" over three players:

| Player | Goals | Assists | Games |
| --- | --- | --- | --- |
| A | 30 | 40 | 80 |
| B | 10 | 20 | 70 |
| C | 20 | 30 | 75 |
| **Average** | **20** | **30** | **75** |

The average vector sits in the middle of the group, like a centre of gravity. Recommendation systems sometimes represent your taste as the average of the vectors of everything you liked. An LLM can represent a whole sentence by averaging (or otherwise combining) its word vectors.

### 3.9 Unit vectors: direction only

Sometimes you care about which way a vector points but not how long it is. Divide the vector by its own length and you get a vector of length exactly 1 pointing the same way, called a **unit vector**. This is called **normalizing**.

$$
\hat{\mathbf{v}} = \frac{\mathbf{v}}{\|\mathbf{v}\|} \qquad \text{e.g.} \quad \frac{[3, 4]}{5} = [0.6, 0.8]
$$

Why throw away length? Imagine comparing two book reviews by counting positive and negative words. A long review has bigger counts than a short one, even if both are equally enthusiastic. Normalizing removes the "how long was the review" effect and keeps only the "what was the attitude" direction. Chapter 4 uses this idea to define cosine similarity, the most common way to compare embeddings.

### 3.10 Where vectors live inside an LLM

Now you can read a precise description of the first half of an LLM:

1. Each token is turned into a vector of *d* numbers, where *d* is the model's **hidden size** (often 2,048 to 16,384).
2. A sentence of *n* tokens becomes *n* vectors, one per token.
3. Each layer updates each vector by adding adjustments to it (vector addition).
4. Attention builds those adjustments as weighted mixtures of other tokens' vectors (scaling and adding).
5. At the end, the last token's vector is compared with every vocabulary word to decide what comes next.

Every one of those steps uses only the operations in this chapter: adding, scaling, mixing and comparing lists of numbers.

> **Developer note.** In Python, the NumPy library makes vector arithmetic look just like the math:
>
> ```python
> import numpy as np
> a = np.array([3, 1]); b = np.array([1, 3])
> a + b        # array([4, 4])
> 2 * a        # array([6, 2])
> np.linalg.norm(a - b)   # distance ≈ 2.83
> ```

> **Common confusion.** "A vector is an arrow" and "a vector is a list" are not two different things. They are two pictures of the same object. Use the arrow picture to build intuition and the list picture to compute.

### Chapter summary

| Operation | Recipe | Everyday example | LLM use |
| --- | --- | --- | --- |
| Add | Add position by position | Combining grocery baskets | Residual connections |
| Subtract | Subtract position by position | Change from last month | Measuring differences in meaning |
| Scale | Multiply every position by one number | Doubling a recipe | Weighting information |
| Weighted mix | Scale several, then add | Blending smoothies | Attention output |
| Length | Square, add, square root | Distance as the crow flies | Normalization |
| Distance | Length of the difference | Song recommendations | Finding similar words |

> **Try it yourself.**
>
> 1. Add \[2, 5, −1\] and \[3, −2, 4\].
> 2. Find the length of \[1, 2, 2\].
> 3. Which is closer to \[1, 1\]: \[2, 2\] or \[1, 3\]?
> 4. Make a 25% / 75% mixture of \[10, 0\] and \[0, 10\].
