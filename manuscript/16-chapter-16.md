## Chapter 16 — Training a Large Language Model

A chat assistant is built in three stages: **pretraining** on a huge amount of text to predict the next token, **supervised fine-tuning** on example conversations, and **preference tuning** from judgements of which answers are better. Every stage uses the math you already know: cross-entropy loss, backpropagation and gradient descent. This chapter shows what changes between the stages, and why scale matters so much.

### 16.1 The three stages

![Figure 16.1 · The three stages of building a chat assistant](figures/fig-16-01.png)

A helpful analogy is training a doctor. First, years of reading textbooks and journals (pretraining): broad knowledge, but no bedside manner. Then a residency where experienced doctors demonstrate how to handle patients (supervised fine-tuning). Finally, ongoing feedback on which decisions were better or worse (preference tuning).

### 16.2 Pretraining data

Pretraining needs text, and lots of it: often several trillion tokens for a modern model. Typical sources include:

- Public web pages, heavily filtered.
- Books and long-form writing.
- Source code from public repositories.
- Scientific papers, encyclopedias and reference works.
- Text in many languages.

Raw web text is messy, so much of the work is cleaning: removing duplicates (the same article copied across 1,000 sites would otherwise be memorized), filtering spam and low-quality pages, removing personal information, and balancing the mix of sources. Data quality matters as much as quantity; a smaller, cleaner dataset can beat a larger, dirtier one.

### 16.3 The training objective: next-token prediction everywhere

The beauty of pretraining is that it needs **no labels**. The text is its own answer key. Every position in every document is a little exam question: "given the tokens so far, what comes next?" This is called **self-supervised learning**.

One sentence becomes many training examples at once:

| Model sees (input) | Must predict (target) |
| --- | --- |
| The | cat |
| The cat | sat |
| The cat sat | on |
| The cat sat on | the |
| The cat sat on the | mat |

Thanks to the causal mask (Chapter 14), all five predictions are made in a single forward pass, and their cross-entropy losses (Chapter 9) are averaged. With a context window of thousands of tokens and batches of millions of tokens, each training step contains millions of these little exams.

To get good at this game across every kind of text, a model must learn a remarkable amount. Predicting the next word of a physics textbook requires some physics. Predicting the next line of a Python program requires understanding the program. Predicting a character's reply in a novel requires modelling what that character wants. Next-token prediction sounds humble, but doing it *well* demands broad knowledge.

### 16.4 What pretraining produces: a base model

The result of pretraining is a **base model**: an extraordinarily good text continuer, but not an assistant. Ask a base model "What is the capital of France?" and it might continue with:

> What is the capital of Germany? What is the capital of Italy? What is the capital of Spain?

because on the web, a question like that often appears in a list of quiz questions. The base model is not being unhelpful; it is doing exactly what it was trained to do, continuing the text in the most likely way. Turning it into an assistant is the job of the next two stages.

### 16.5 How much computing does it take?

A handy rule of thumb: training a model with N parameters on D tokens takes about 6 × N × D arithmetic operations. (Roughly 2 per parameter per token for the forward pass, and 4 for the backward pass.)

$$
\text{training compute} \approx 6 \, N \, D
$$

**Worked example.** A 7-billion-parameter model trained on 2 trillion tokens:

- 6 × (7 × 10⁹) × (2 × 10¹²) = 8.4 × 10²² operations.
- A data-centre GPU such as the NVIDIA A100 does, in practice, roughly 1.25 × 10¹⁴ useful operations per second (about 40% of its peak).
- Time: 8.4 × 10²² / 1.25 × 10¹⁴ ≈ 6.7 × 10⁸ GPU-seconds ≈ **187,000 GPU-hours**.

That is about 21 years on one GPU, or roughly 8 days on 1,000 GPUs running in parallel. The largest models today use many times more. This is why training frontier models costs tens or hundreds of millions of dollars, while running them for one question costs a fraction of a cent.

### 16.6 Scaling laws: bigger is predictably better

One of the most important discoveries in modern AI is that loss improves **predictably** as you scale up parameters, data and compute together. Plot loss against compute on a logarithmic axis and you get a smooth curve:

![Figure 16.2 · Illustrative curve showing the typical shape of published scaling results](figures/fig-16-02.png)

This is called a **scaling law**, and it has the shape of a **power law**: each tenfold increase in compute buys a steady but gradually smaller improvement. Because the curve is so smooth, labs can train small, cheap models, fit the curve, and predict how good a giant model will be before spending the money to train it.

A well-known study in 2022 (the "Chinchilla" paper from DeepMind) found that for a fixed compute budget, it is best to grow parameters and data together, with roughly **20 training tokens per parameter**. Many earlier models had been too big for their data. Since then, many labs train well past that ratio, accepting extra training cost to get a smaller model that is cheaper to run.

> **Common confusion.** A lower loss does not translate neatly into "smarter". Loss measures average next-token surprise. But in practice, lower loss has gone together with new abilities: following instructions, writing working code, multi-step reasoning. Some abilities appear fairly suddenly as models grow, which is one of the open research puzzles of the field.

### 16.7 Supervised fine-tuning: learning by example

To make a base model behave like an assistant, it is shown many example conversations, written or curated by people, in a fixed format:

> **User:** What is the capital of France? **Assistant:** The capital of France is Paris.

The training is the same next-token prediction with the same cross-entropy loss, with one tweak: the loss is computed only on the assistant's words, not the user's. The model is learning to *produce* good replies, not to predict what users will type. After this stage, the model reliably answers questions instead of continuing quiz lists. This stage is called **supervised fine-tuning** (SFT) or **instruction tuning**.

### 16.8 Preference tuning: learning from comparisons

It is hard to write the perfect answer to every question, but it is much easier to look at two answers and say which is better. Preference tuning exploits this.

**Step 1: collect comparisons.** For many prompts, the model produces two answers. People (and, increasingly, AI models following written guidelines) judge which is better: more helpful, more accurate, safer.

**Step 2: train a reward model.** A separate network learns to give each answer a score r. It is trained so that preferred answers get higher scores, using a formula you can now read:

$$
P(A \text{ preferred over } B) = \sigma(r_A - r_B) = \frac{1}{1 + e^{-(r_A - r_B)}}
$$

That is the sigmoid from Chapter 13 applied to the *difference* in scores, just like softmax turning differences into ratios in Chapter 8. If A's score is 2 higher than B's, the model predicts A is preferred with probability σ(2) ≈ 0.88. The reward model is trained with cross-entropy loss on these predictions.

**Step 3: optimize the assistant.** The assistant generates answers, the reward model scores them, and the assistant's weights are nudged toward answers that score higher. This is **reinforcement learning from human feedback (RLHF)**. A penalty based on KL divergence (Chapter 9) keeps the new model close to the fine-tuned one, so it does not drift into weird text that happens to fool the reward model. Think of it as a leash.

A popular simpler alternative, **direct preference optimization (DPO)**, skips the separate reward model and adjusts the assistant directly on the comparison pairs, using a loss built from the same sigmoid-of-differences idea.

### 16.9 The engineering behind the math

The math of training is what you have learned. Doing it at scale adds engineering challenges:

| Challenge | What is done about it |
| --- | --- |
| The model does not fit on one GPU | Split the matrices and layers across many GPUs (model parallelism) |
| One GPU is too slow | Give thousands of GPUs different mini-batches and average their gradients (data parallelism) |
| Memory and speed | Store numbers in 16-bit formats such as bfloat16 instead of 32-bit (Chapter 18) |
| Hardware failures over weeks of training | Save checkpoints regularly and resume after crashes |
| Training instability (loss spikes) | Careful learning rate schedules, normalization and gradient clipping |

### Chapter summary

- Pretraining teaches next-token prediction on trillions of tokens; the text is its own label.
- A base model continues text; it is not yet an assistant.
- Training compute ≈ 6 × parameters × tokens, which for real models means hundreds of thousands of GPU-hours or more.
- Scaling laws show loss falling smoothly and predictably with more compute.
- Supervised fine-tuning teaches the assistant format from examples.
- Preference tuning (RLHF or DPO) learns from comparisons, using a sigmoid of score differences and a KL leash.

> **Try it yourself.**
>
> 1. Estimate the training compute for a 1-billion-parameter model trained on 20 billion tokens.
> 2. Two answers have reward scores 1.0 and 0.0. What probability does the reward model give that the first is preferred?
> 3. Write five (input, target) training pairs from the sentence "Practice makes progress, not perfection."
