## Chapter 5 — Matrices: Machines That Transform Vectors

A matrix is a rectangular table of numbers, and multiplying a vector by a matrix transforms it into a new vector. Almost all the "thinking" inside an LLM is matrix multiplication: more than 90% of the computing time of a typical transformer is spent on it. This chapter teaches you to do it by hand and, more importantly, to see what it does.

### 5.1 A matrix is a table

You have seen many matrices. A spreadsheet of numbers is one. So is a nutrition table:

|  | Egg | Toast | Banana |
| --- | --- | --- | --- |
| Calories | 78 | 80 | 105 |
| Protein (g) | 6 | 3 | 1 |

Stripped of its labels, that table is a matrix with 2 rows and 3 columns:

$$
N = \begin{bmatrix} 78 & 80 & 105 \\ 6 & 3 & 1 \end{bmatrix}
$$

We describe a matrix by its **shape**: rows first, then columns. This one is a "2 by 3" matrix, written 2 × 3. Matrices are usually named with capital letters, and the number in row *i*, column *j* is written Nᵢⱼ. So N₁₃ = 105 (row 1, column 3).

A vector is just a matrix with one column (or one row). Seen this way, everything in linear algebra is tables of numbers.

### 5.2 Matrix times vector: many dot products at once

For breakfast you eat 2 eggs, 1 toast and 1 banana: the vector \[2, 1, 1\]. How many calories and grams of protein is that?

For calories, take the calories row and dot it with your servings: 78 × 2 + 80 × 1 + 105 × 1 = 341. For protein, take the protein row and do the same: 6 × 2 + 3 × 1 + 1 × 1 = 16.

![Figure 5.1 · Approximate nutrition values, rounded](figures/fig-05-01.png)

That is matrix-vector multiplication. **Each row of the matrix takes a dot product with the vector, and each result becomes one entry of the output.**

$$
\begin{bmatrix} 78 & 80 & 105 \\ 6 & 3 & 1 \end{bmatrix} \begin{bmatrix} 2 \\ 1 \\ 1 \end{bmatrix} = \begin{bmatrix} 341 \\ 16 \end{bmatrix}
$$

Three rules follow from this recipe:

1. The number of **columns** of the matrix must equal the **length** of the vector (3 foods, 3 servings).
2. The output has one entry per **row** of the matrix (2 nutrients, 2 totals).
3. A 2 × 3 matrix therefore turns a 3-number vector into a 2-number vector.

> **Try it yourself.** Lunch is 1 egg, 2 toasts and 0 bananas. Multiply the nutrition matrix by \[1, 2, 0\]. (Answer: \[238, 12\].)

### 5.3 Seeing a matrix as a machine

The nutrition example treats the matrix as a lookup table. The deeper view is that **a matrix is a function**: put a vector in, get a vector out. And it is a special kind of function, called **linear**, which means it respects adding and scaling: double the input and the output doubles; add two inputs and the outputs add.

In two dimensions you can watch what a matrix does by applying it to every point of a shape.

![Figure 5.2 · One 2 × 2 matrix acting on a unit square](figures/fig-05-02.png)

This particular matrix stretched the square sideways and slanted it. Other matrices do other things:

| Matrix | What it does to every arrow | Everyday analogy |
| --- | --- | --- |
| \[\[2, 0\], \[0, 2\]\] | Doubles its length | Zooming in on a photo |
| \[\[0, −1\], \[1, 0\]\] | Rotates it 90° | Turning a map |
| \[\[1, 0\], \[0, 0\]\] | Flattens it onto the horizontal axis | A shadow at noon |
| \[\[1, 0\], \[0, 1\]\] | Leaves it unchanged | A photocopy |

The last one, with 1s on the diagonal and 0s elsewhere, is called the **identity matrix**. It is the "do nothing" machine, just as multiplying by 1 does nothing to a number.

> **LLM connection.** Every layer of an LLM contains matrices that move token vectors around in this way, rotating, stretching and projecting them, so that useful information becomes easy to read off. The matrices are learned: training discovers which transformations make next-word prediction easier.

### 5.4 Matrices can change the number of dimensions

The nutrition matrix took 3 numbers (foods) and produced 2 (nutrients). A matrix with more columns than rows **compresses**: it summarizes many numbers into fewer. A matrix with more rows than columns **expands**: it spreads a few numbers out into more.

Think of a company report. Hundreds of daily sales figures go in; a handful of summary figures (total, average per store, growth) come out. That is compression. Expansion is like a chef turning three basic ingredients into a ten-item tasting menu, each dish a different combination.

Transformers use both:

- Attention projects each 4,096-number token vector down to small 128-number vectors to compare words cheaply.
- The feed-forward layer expands each vector to around 14,000 numbers, works on it there, then compresses it back to 4,096.

### 5.5 Matrix times matrix: many vectors at once

What if you want the totals for breakfast, lunch and dinner? You could do three matrix-vector multiplications. Or you could put the three meals side by side as the columns of a second matrix and multiply once:

$$
\begin{bmatrix} 78 & 80 & 105 \\ 6 & 3 & 1 \end{bmatrix} \begin{bmatrix} 2 & 1 & 0 \\ 1 & 2 & 1 \\ 1 & 0 & 2 \end{bmatrix} = \begin{bmatrix} 341 & 238 & 290 \\ 16 & 12 & 5 \end{bmatrix}
$$

Each column of the result is one meal's totals. The rule for each cell: **the cell in row i, column j of the result is the dot product of row i of the first matrix with column j of the second.**

Let's check one cell: row 1 (calories) with column 3 (dinner: 0 eggs, 1 toast, 2 bananas) gives 78 × 0 + 80 × 1 + 105 × 2 = 290. It matches.

This is why matrix multiplication is so central to LLMs. A sentence of 5 tokens is 5 vectors. Stack them into a matrix and a single matrix multiplication transforms all 5 at once. A batch of 1,000 sentences can be processed in one enormous multiplication.

### 5.6 The shape rule

Shapes are the most common source of bugs when people write neural-network code, so learn this rule well:

$$
(n \times d) \cdot (d \times m) = (n \times m)
$$

![Figure 5.3 · The shape rule for matrix multiplication](figures/fig-05-03.png)

The two inner numbers must match, and they "cancel". The two outer numbers give the shape of the answer. If the inner numbers do not match, the multiplication is impossible, just as you cannot price 4 items with 3 prices.

| First | Second | Result | Allowed? |
| --- | --- | --- | --- |
| 2 × 3 | 3 × 1 | 2 × 1 | Yes |
| 5 × 4 | 4 × 3 | 5 × 3 | Yes |
| 10 × 4,096 | 4,096 × 4,096 | 10 × 4,096 | Yes |
| 3 × 2 | 3 × 2 | — | No: inner 2 and 3 differ |

> **Common confusion.** For ordinary numbers, 3 × 5 = 5 × 3. For matrices, the order matters: A × B is usually different from B × A, and often one of them is not even allowed. Matrix multiplication is like composing functions: rotate then stretch is not the same as stretch then rotate.

### 5.7 The transpose: flipping rows and columns

The **transpose** of a matrix swaps its rows and columns. It is written with a small T:

$$
\begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix}^{T} = \begin{bmatrix} 1 & 4 \\ 2 & 5 \\ 3 & 6 \end{bmatrix}
$$

It is like rotating a spreadsheet so that the rows become columns. A 2 × 3 matrix becomes 3 × 2. Transposes appear in the attention formula as Kᵀ, where flipping the matrix K lines up its shape so that every query can be dotted with every key in one multiplication (Chapter 14).

### 5.8 Counting parameters

Every number inside a weight matrix is one **parameter**: one adjustable knob the model learns. Counting them is just multiplying the shape.

| Matrix in one layer of a mid-size model | Shape | Parameters |
| --- | --- | --- |
| Query projection | 4,096 × 4,096 | 16.8 million |
| Key projection | 4,096 × 4,096 | 16.8 million |
| Value projection | 4,096 × 4,096 | 16.8 million |
| Output projection | 4,096 × 4,096 | 16.8 million |
| Feed-forward up | 4,096 × 11,008 | 45.1 million |
| Feed-forward down | 11,008 × 4,096 | 45.1 million |

That is roughly 157 million parameters in one layer, before smaller pieces. Stack 32 layers and you are past 5 billion. Add the embedding table and the third feed-forward matrix many models use (see the note below) and you reach about 7 billion, the size of many popular open models. Nothing mysterious makes a model "large": it is big matrices, many times over.

> **Note.** Some models use extra tricks (such as a third feed-forward matrix, or sharing key and value matrices across heads), so exact counts differ. The method of counting, shape times shape, is always the same.

### 5.9 Why GPUs love matrices

A matrix multiplication of two 4,096 × 4,096 matrices needs about 69 billion multiply-and-add steps. That sounds slow, but each output cell is an independent dot product: none waits for any other. A graphics card (GPU) contains thousands of small calculators that can each work on different cells at the same time, like thousands of cashiers each ringing up a different customer.

This is why the rise of LLMs went hand in hand with the rise of GPUs. The math of transformers is almost entirely matrix multiplication, and GPUs are machines built to do matrix multiplication fast.

> **Developer note.** In NumPy or PyTorch, the `@` operator multiplies matrices:
>
> ```python
> N = np.array([[78, 80, 105], [6, 3, 1]])
> meals = np.array([[2, 1, 0], [1, 2, 1], [1, 0, 2]])
> N @ meals   # array([[341, 238, 290], [16, 12, 5]])
> N.shape, meals.shape, (N @ meals).shape   # (2, 3) (3, 3) (2, 3)
> ```

### Chapter summary

- A matrix is a table of numbers with a shape: rows × columns.
- Matrix × vector = each row dotted with the vector.
- A matrix is a linear function: it moves, stretches, rotates or projects every vector.
- Matrices can compress (fewer outputs than inputs) or expand (more outputs).
- Matrix × matrix transforms many vectors at once; inner sizes must match.
- Every number in a weight matrix is a learned parameter; LLMs are built from billions of them.

> **Try it yourself.**
>
> 1. Multiply \[\[1, 2\], \[3, 4\]\] by the vector \[5, 6\].
> 2. What shape do you get from (8 × 512) times (512 × 64)?
> 3. A layer turns 768-number vectors into 3,072-number vectors. How many weights does its matrix hold?
