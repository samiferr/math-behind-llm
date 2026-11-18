## Chapter 18 — Putting It All Together

You now know every piece of math an LLM uses. This final chapter follows one sentence through a complete model, adding up the shapes and operations, then explains how numbers are stored inside the computer and why shrinking them (quantization) lets big models run on small machines. It ends with a map of what you have learned and where to go next.

### 18.1 One sentence, start to finish

The promise in the preface was to follow "The cat sat on the" through a model and explain, with real math, why it says "mat". Here it is, for a model shaped like a 7-billion-parameter open model:

![Figure 18.1 · Shapes for a model with d = 4,096, 32 blocks and a 128,000-token vocabulary](figures/fig-18-01.png)

Step by step:

1. **Tokenize (Chapter 6).** The tokenizer splits the text into 5 tokens and looks up their IDs. These are just labels, with no meaning yet.
2. **Embed (Chapters 3, 5, 6).** Each ID selects one row of the 128,000 × 4,096 embedding table. We now have a 5 × 4,096 matrix: five vectors, one per token, each a learned "profile" of its word.
3. **Thirty-two transformer blocks (Chapters 13 to 15).** In each block, every token's vector is normalized, then:
   - multiplied by W\_Q, W\_K and W\_V to make queries, keys and values;
   - each query is dotted with the keys of itself and all earlier tokens, scaled by √128, and softmaxed into weights (with rotary position encoding making those scores sensitive to distance);
   - the values are mixed by those weights, the heads are concatenated, multiplied by W\_O, and **added** back to the stream;
   - the result is normalized again, passed through the feed-forward network (expand, activate, compress), and **added** back again. After 32 blocks, the vector for the last "the" no longer just means "the". Through attention it has gathered "cat", "sat" and "on", and through the feed-forward layers it has turned that into something like "the next word is a surface a cat might sit on".
4. **Last vector (Chapter 14).** Because of the causal mask, the final token's vector is the only one that has seen the whole sentence, so it is the one used to predict what comes next.
5. **Score every word (Chapters 4, 5).** That 4,096-number vector is multiplied by the 4,096 × 128,000 unembedding matrix: 128,000 dot products, one per vocabulary token. The vector for "mat" agrees strongly with it, the vector for "democracy" barely at all.
6. **Probabilities (Chapters 2, 8, 17).** Softmax turns the 128,000 logits into probabilities; temperature and top-p reshape and trim them.
7. **Choose (Chapters 7, 17).** A random draw picks "mat" with, say, 59% chance. It is appended, and the whole process runs again for the next token, now reusing the cached keys and values.

And why "mat" specifically? Because during pretraining (Chapter 16), every time the model saw a phrase like "the cat sat on the mat" and gave "mat" a low probability, the cross-entropy loss (Chapter 9) was high, backpropagation (Chapter 11) found how every weight contributed, and gradient descent (Chapter 12) nudged them all so that next time "mat" would score a bit higher. Repeat over trillions of tokens, and the weights come to encode the patterns of language, and much else besides.

### 18.2 How much arithmetic per word?

A useful rule: generating one token takes about **2 × N** arithmetic operations, where N is the number of parameters (one multiply and one add per weight). For a 7-billion-parameter model, that is about 14 billion operations per token. A 500-token answer needs around 7 trillion operations. A modern GPU does that in seconds.

Notice that training (Chapter 16) needs about 6 × N per token, three times as much as generation, because of the backward pass, and it is repeated for trillions of tokens. That is why training is so much more expensive than using a model.

### 18.3 How computers store numbers

Every weight is stored as a **floating-point** number, the computer's version of scientific notation: a few significant digits plus an exponent, like 6.02 × 10²³. The more bits you use, the more precise the number.

| Format | Bits per number | Roughly how precise | Typical use |
| --- | --- | --- | --- |
| FP32 | 32 | About 7 significant digits | Traditional default; optimizer bookkeeping |
| FP16 | 16 | About 3 digits, limited range | Older mixed-precision training |
| BF16 | 16 | About 2 to 3 digits, same range as FP32 | Standard for LLM training today |
| INT8 / INT4 | 8 / 4 | 256 / 16 possible values per scale | Running models (inference) |

Think of rulers. A ruler with millimetre marks (FP32) is precise. One with only centimetre marks (BF16) is coarser but lighter to carry. For neural networks, the coarse ruler turns out to be good enough most of the time: models are surprisingly tolerant of small rounding errors, because each output depends on thousands of weights whose errors tend to cancel out.

### 18.4 Quantization: shrinking a model

The weights of a model must sit in fast memory while it runs. Fewer bits per weight means less memory and faster movement of data through the chip:

![Figure 18.2 · Weight memory = parameters × bits ÷ 8, computed](figures/fig-18-02.png)

**Quantization** converts weights to a small set of integer levels. A simple 8-bit recipe for a group of weights:

1. Find the largest absolute value in the group. Suppose the weights are \[−0.82, 0.10, 0.45, 1.20\]; the largest is 1.20.
2. Choose a **scale** so that this maps to 127, the biggest 8-bit value: scale = 1.20 / 127 ≈ 0.00945.
3. Divide each weight by the scale and round to the nearest whole number:

| Original weight | Weight ÷ scale | Stored integer | Restored (integer × scale) | Error |
| --- | --- | --- | --- | --- |
| −0.82 | −86.8 | −87 | −0.822 | 0.002 |
| 0.10 | 10.6 | 11 | 0.104 | 0.004 |
| 0.45 | 47.6 | 48 | 0.454 | 0.004 |
| 1.20 | 127.0 | 127 | 1.200 | 0.000 |

Each weight now takes 1 byte instead of 2 or 4, and the errors are tiny. Real methods are cleverer (separate scales for small groups of weights, special handling for rare large values, calibration on sample text), and 4-bit versions of models often perform close to the original. This is how people run capable open models on laptops and phones.

### 18.5 The map of what you learned

| Math idea | Chapter | Where it lives in an LLM |
| --- | --- | --- |
| Functions and composition | 2 | The model is a chain of functions |
| Exponentials and logarithms | 2 | Softmax; log-probabilities; loss |
| Vectors, adding and scaling | 3 | Embeddings; residual stream; mixing values |
| Dot product and cosine similarity | 4 | Attention scores; neurons; output logits; search |
| Matrices and matrix multiplication | 5 | Every weight layer; most of the computation |
| Embeddings | 6 | Turning tokens into meaningful vectors |
| Conditional probability and chain rule | 7 | P(next token given context); generation |
| Softmax and temperature | 8 | Output probabilities; attention weights |
| Entropy and cross-entropy | 9 | The training loss; perplexity |
| Derivatives | 10 | Sensitivity of the loss to each weight |
| Gradients and the chain rule | 11 | Backpropagation |
| Gradient descent and Adam | 12 | How weights are updated |
| Neurons and activations | 13 | Feed-forward networks |
| Attention | 14 | How tokens share information |
| Normalization and residuals | 15 | Stable, deep networks |
| Scaling, fine-tuning, preferences | 16 | How base models become assistants |
| Sampling | 17 | How text is produced |
| Floating point and quantization | 18 | Running models efficiently |

### 18.6 What this book left out

To stay beginner-friendly, this book skipped several active topics. Each builds directly on what you now know:

- **Mixture of experts.** Instead of one feed-forward network per block, the model has many "expert" networks and a small router that sends each token to only a few of them. More parameters, similar cost per token.
- **Long-context techniques.** Ways to make attention cheaper or to stretch positional encodings so that models can read hundreds of thousands of tokens.
- **Multimodal models.** Images and audio are cut into patches or frames, turned into vectors, and fed into the same transformer alongside text tokens.
- **Reasoning models.** Models trained to write out intermediate steps ("chain of thought") before answering, spending more computation at answer time on harder problems.
- **Interpretability.** Research that opens up trained models to find which directions, neurons and attention heads represent which concepts.
- **Agents and tool use.** Models that call search engines, run code or use other software, with their outputs fed back into the context.

### 18.7 Where to go next

The best next step is to **build a tiny model yourself**. Reading about attention is good; watching your own 10-million-parameter model learn to produce Shakespeare-like text is unforgettable. The appendix lists resources in a sensible order. A good path:

1. Strengthen intuition with visual explanations of linear algebra and calculus.
2. Code a tiny neural network from scratch, including backpropagation, in plain Python.
3. Build a small GPT-style model in PyTorch and train it on a book's worth of text.
4. Read the original transformer paper; you will now understand nearly every line.
5. Pick one area from 18.6 that excites you and go deep.

### 18.8 A closing thought

When you started this book, "large language model" may have sounded like magic. You now know it is a chain of very ordinary operations: look up some vectors, multiply some matrices, take some dot products, apply softmax, add things up, repeat. And you know how the numbers in those matrices were found: by measuring surprise and walking downhill, a tiny step at a time, trillions of times.

That the result can write a sonnet, explain a tax rule or debug your code is genuinely remarkable. But it is remarkable in the way a cathedral is remarkable, built from ordinary stones arranged with great care at enormous scale, not in the way a magic trick is. You now know what the stones are. Go build something.

> **Try it yourself.**
>
> 1. How many operations does a 70-billion-parameter model need to generate 1,000 tokens?
> 2. How much memory do the weights of a 13-billion-parameter model need in BF16? In INT4?
> 3. Quantize the weights \[0.5, −0.25, 1.0\] to 8 bits using the recipe in 18.4.
