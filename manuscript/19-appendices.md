# Appendices

## Appendix A — Notation Cheat Sheet

| Symbol | Read as | Meaning | First seen |
| --- | --- | --- | --- |
| f(x) | "f of x" | A function applied to input x | Ch. 2 |
| f(g(x)) | "f of g of x" | Composition: apply g, then f | Ch. 2 |
| e | "e" | Euler's number, about 2.718 | Ch. 2 |
| eˣ, exp(x) | "e to the x" | Exponential function; always positive | Ch. 2 |
| ln(x) | "natural log of x" | Undoes eˣ; turns products into sums | Ch. 2 |
| log₂(x) | "log base 2 of x" | How many doublings make x; used for bits | Ch. 2, 9 |
| Σ | "sum" | Add up a list | Ch. 2 |
| Π | "product" | Multiply a list | Ch. 7 |
| **v**, \[3, 1\] | "vector v" | An ordered list of numbers | Ch. 3 |
| vᵢ | "v sub i" | The i-th component of **v** | Ch. 3 |
| ‖**v**‖ | "length of v" | Magnitude (norm) of a vector | Ch. 3 |
| **a** · **b** | "a dot b" | Dot product: multiply matching entries, add | Ch. 4 |
| θ | "theta" | An angle | Ch. 4 |
| W, A | capital letters | Matrices | Ch. 5 |
| Wᵀ | "W transpose" | Rows and columns swapped | Ch. 5 |
| n × d | "n by d" | Shape: n rows, d columns | Ch. 5 |
| V | "V" | Vocabulary size | Ch. 6 |
| d | "d" | Hidden size (vector length) | Ch. 6 |
| P(A) | "probability of A" | A number from 0 to 1 | Ch. 7 |
| P(A \| B) | "P of A given B" | Probability of A when B is known | Ch. 7 |
| z | "z" | A logit (raw score) | Ch. 8 |
| T | "temperature" | Divides logits before softmax | Ch. 8 |
| H(P) | "entropy of P" | Average surprise of a distribution | Ch. 9 |
| H(P, Q) | "cross-entropy" | Average surprise under beliefs Q when truth is P | Ch. 9 |
| D\_KL | "KL divergence" | Extra surprise from wrong beliefs | Ch. 9 |
| Δ | "delta" | A change in | Ch. 10 |
| f′(x), dy/dx | "f prime", "dee y dee x" | Derivative: rate of change | Ch. 10 |
| ∂L/∂w | "partial L partial w" | Partial derivative: change one input only | Ch. 11 |
| ∇L | "nabla L", "gradient of L" | Vector of all partial derivatives | Ch. 11 |
| η | "eta" | Learning rate | Ch. 12 |
| σ(x) | "sigma of x" | Sigmoid function, 1 / (1 + e⁻ˣ) | Ch. 13 |
| Q, K, V | "queries, keys, values" | The three projections used in attention | Ch. 14 |
| αᵢⱼ | "alpha i j" | How much token i attends to token j | Ch. 14 |
| μ, σ | "mu", "sigma" | Mean and standard deviation | Ch. 15 |
| γ, β | "gamma", "beta" | Learned scale and shift in normalization | Ch. 15 |
| N, D | "N", "D" | Parameter count, training tokens | Ch. 16 |

## Appendix B — The Key Formulas on One Page

**Weighted sum (a neuron before activation)**

$$
z = \sum_i w_i x_i + b = \mathbf{w} \cdot \mathbf{x} + b
$$

**Vector length and cosine similarity**

$$
\|\mathbf{v}\| = \sqrt{\sum_i v_i^2} \qquad \cos(\mathbf{a}, \mathbf{b}) = \frac{\mathbf{a} \cdot \mathbf{b}}{\|\mathbf{a}\| \, \|\mathbf{b}\|}
$$

**Matrix shape rule**

$$
(n \times d) \cdot (d \times m) = (n \times m)
$$

**Chain rule of probability**

$$
P(w_1, \dots, w_n) = \prod_{t=1}^{n} P(w_t \mid w_1, \dots, w_{t-1})
$$

**Softmax with temperature**

$$
P_i = \frac{e^{z_i / T}}{\sum_j e^{z_j / T}}
$$

**Language-model loss and perplexity**

$$
\text{loss} = -\frac{1}{n} \sum_{t=1}^{n} \ln P(w_t \mid w_{<t}) \qquad \text{perplexity} = e^{\text{loss}}
$$

**Chain rule of calculus**

$$
\frac{dL}{dw} = \frac{dL}{dh} \cdot \frac{dh}{dw}
$$

**Gradient descent**

$$
\mathbf{w} \leftarrow \mathbf{w} - \eta \, \nabla L(\mathbf{w})
$$

**A layer of neurons**

$$
\mathbf{h} = f(W \mathbf{x} + \mathbf{b})
$$

**Scaled dot-product attention**

$$
\text{Attention}(Q, K, V) = \text{softmax}\!\left(\frac{Q K^{T}}{\sqrt{d_k}}\right) V
$$

**Layer normalization**

$$
\text{LayerNorm}(\mathbf{x}) = \gamma \cdot \frac{\mathbf{x} - \mu}{\sigma} + \beta
$$

**Rules of thumb**

$$
\text{training compute} \approx 6ND \qquad \text{generation compute per token} \approx 2N \qquad \text{weight memory} = N \times \frac{\text{bits}}{8}
$$

## Appendix C — Glossary

| Term | Plain-language meaning | Chapter |
| --- | --- | --- |
| Activation function | A simple bending function applied after a weighted sum (ReLU, GELU, sigmoid) | 13 |
| Adam / AdamW | Gradient descent with momentum and a separate step size for each weight | 12 |
| Attention | Each token builds a new vector as a weighted mix of other tokens' values | 14 |
| Attention head | One independent attention operation; models run many in parallel | 14 |
| Autoregressive | Generating one token at a time, each based on all previous ones | 14, 17 |
| Backpropagation | Computing every weight's gradient by applying the chain rule backward from the loss | 11 |
| Base model | A pretrained model that continues text but is not yet tuned as an assistant | 16 |
| Batch (mini-batch) | The sample of training examples used for one gradient step | 12 |
| Bias | A learned number added after a weighted sum; the "intercept" | 2, 13 |
| Bit | The information in one fair yes/no answer | 9 |
| Causal mask | Blocks each token from attending to later tokens | 14 |
| Chain rule (calculus) | Sensitivities through a chain of functions multiply | 11 |
| Chain rule (probability) | A sequence's probability is the product of next-item probabilities | 7 |
| Context window | The maximum number of tokens a model can consider at once | 14 |
| Cosine similarity | How closely two vectors point the same way, from −1 to 1 | 4 |
| Cross-entropy | Average surprise when your beliefs differ from reality; the LLM loss | 9 |
| Derivative | Rate of change; slope of a curve at a point | 10 |
| Dot product | Multiply matching entries of two vectors and add: an agreement score | 4 |
| Embedding | A learned vector representing a token | 6 |
| Entropy | Average surprise; how uncertain a distribution is | 9 |
| Epoch | One full pass over the training data | 12 |
| Feed-forward network | The per-token neural network in each transformer block | 13, 15 |
| Fine-tuning | Further training of a pretrained model on a narrower dataset | 16 |
| Gradient | Vector of partial derivatives; points steepest uphill | 11 |
| Gradient descent | Repeatedly stepping against the gradient to reduce the loss | 12 |
| Greedy decoding | Always choosing the most likely next token | 17 |
| Hallucination | Fluent, confident output that is false | 17 |
| Hidden size (d) | The length of each token's vector inside the model | 6, 15 |
| KL divergence | Extra surprise caused by beliefs Q differing from truth P | 9, 16 |
| KV cache | Stored keys and values of earlier tokens, reused during generation | 14, 17 |
| Layer normalization | Rescaling a vector to mean 0 and spread 1, then a learned scale and shift | 15 |
| Learning rate | The step size in gradient descent | 12 |
| Logit | A raw, unnormalized score before softmax | 8 |
| Loss | A single number measuring how wrong the model's predictions are | 9 |
| Matrix | A rectangular table of numbers; a linear transformation | 5 |
| Overfitting | Memorizing training data instead of learning general patterns | 12 |
| Parameter | One learned number (weight or bias) in the model | 5 |
| Perplexity | e raised to the loss; the effective number of choices the model is torn between | 9 |
| Positional encoding | Information about token order added to or applied within the model | 15 |
| Pretraining | Training on huge amounts of text to predict the next token | 16 |
| Quantization | Storing weights with fewer bits to save memory and time | 18 |
| Residual connection | Adding a sub-layer's output back onto its input | 15 |
| RLHF | Reinforcement learning from human feedback; tuning on preference comparisons | 16 |
| RoPE | Rotary position embeddings; rotating queries and keys by position | 15 |
| Sampling | Choosing the next token at random in proportion to its probability | 17 |
| Scaling law | The smooth, predictable improvement of loss with more compute, data and parameters | 16 |
| Softmax | Turns logits into probabilities: exponentiate, then divide by the total | 8 |
| Temperature | Divides logits before softmax; low sharpens, high flattens | 8, 17 |
| Token | A piece of text (word or word fragment) the model processes as a unit | 6 |
| Tokenizer | The component that splits text into tokens and maps them to IDs | 6 |
| Top-k / top-p | Keep only the k most likely tokens, or the smallest set reaching probability p | 17 |
| Transformer | The architecture of modern LLMs: stacked blocks of attention and feed-forward layers | 15 |
| Vector | An ordered list of numbers; an arrow with direction and length | 3 |
| Weight | A learned multiplier saying how much an input matters | 2, 13 |

## Appendix D — Answers to "Try it yourself"

**Chapter 2.** Gym: cost = 50 + 30m; slope 30, intercept 50; a steeper start fee and slope than the taxi. (1) f(g(2)) = f(4) = 10 and g(f(2)) = g(4) = 16: not equal. (2) e ≈ 2.718, e² ≈ 7.389, e³ ≈ 20.09, and 2.718 × 7.389 ≈ 20.08. (3) ln(0.9) ≈ −0.105, ln(0.5) ≈ −0.693, ln(0.01) ≈ −4.605; 0.01 is the worst.

**Chapter 3.** (1) \[5, 3, 3\]. (2) √(1 + 4 + 4) = 3. (3) \[2, 2\] is √2 ≈ 1.41 away; \[1, 3\] is 2 away; so \[2, 2\] is closer. (4) \[2.5, 7.5\].

**Chapter 4.** (1) 2 − 4 + 6 = 4. (2) 1 / (1 × √2) ≈ 0.707, which is 45°. (3) First movie: 0.9 + 0.2 − 0.8 = 0.3; second: 0.5 + 0.7 + 0.3 = 1.5. Watch the second.

**Chapter 5.** (1) \[17, 39\]. (2) 8 × 64. (3) 768 × 3,072 = 2,359,296 weights.

**Chapter 6.** (1) dog · sat = −0.32; mat · sat = 0.33; "sat" is more similar to "mat" in this toy table. (2) 32,000 × 4,096 = 131,072,000 numbers.

**Chapter 7.** (1) P(red) = 0.3, P(not red) = 0.7. (2) P(sat | cat) = 0.5 and P(ate | cat) = 0.5. (3) 1 × 0.4 × 0.5 = 0.2.

**Chapter 8.** (1) \[0, 0\] gives 50% each; \[5, 0\] gives the first about 99.3%. (2) e² ≈ 7.4 times. (3) Down: higher temperature flattens the distribution.

**Chapter 9.** (1) −ln(0.25) ≈ 1.39 nats, or exactly 2 bits. (2) log₂(8) = 3 bits. (3) About 10.

**Chapter 10.** (1) 3x² at x = 2 is 12. (2) 2(4 + 1) = 10 is positive, so decrease w. (3) (e⁰·⁰¹ − 1) / 0.01 ≈ 1.005, matching the rule that the slope of eˣ at 0 is e⁰ = 1.

**Chapter 11.** (1) \[2, 6\]. (2) At w = 5, the error is 0, so the gradient is 0: the weight is already perfect. (3) 0.9 × 1.1 × 0.5 ≈ 0.50.

**Chapter 12.** (1) With η = 0.5, w jumps from 0 straight to 3 in one step: for this particular loss, that learning rate is exactly right. (2) With η = 1.0, w goes 0, 6, 0, 6, … bouncing forever without converging. (3) Overfitting; keep the checkpoint with the lowest validation loss and add data or regularization.

**Chapter 13.** (1) Hidden sums \[1, 1\], ReLU gives \[1, 1\], output 2 + 3 + 0.5 = 5.5. (2) For \[3, 1\]: 3 − 2 + 0.5 = 1.5. For \[1, 1\]: −0.5, so ReLU gives 0. (3) 2 × 1,024 × 4,096 = 8,388,608 weights.

**Chapter 14.** (1) Scores 1, 0, −1; scaled 0.71, 0, −0.71; weights about 0.58, 0.28, 0.14. (2) 5: itself and the 4 before it. (3) 2,048 / 16 = 128.

**Chapter 15.** (1) Mean 2, standard deviation √2 ≈ 1.41, result ≈ \[−0.71, −0.71, 1.41\]. (2) Attention adds up weighted values, and sums ignore order; without position information, "dog bites man" and "man bites dog" look identical. (3) 24 of each.

**Chapter 16.** (1) 6 × 10⁹ × 2 × 10¹⁰ = 1.2 × 10²⁰ operations. (2) σ(1) ≈ 0.73. (3) (Practice → makes), (Practice makes → progress), (Practice makes progress → ,), (… progress, → not), (… not → perfection). Real tokenizers may split words differently.

**Chapter 17.** (1) 0.10 → mat, 0.60 → floor, 0.97 → roof. (2) The first two (0.5 + 0.3 = 0.8) survive, rescaled to 0.625 and 0.375. (3) Code has one correct continuation, so the most likely token is best; poetry benefits from variety and surprise.

**Chapter 18.** (1) About 2 × 70 × 10⁹ × 1,000 = 1.4 × 10¹⁴ operations. (2) BF16: 26 GB; INT4: 6.5 GB. (3) Scale = 1.0 / 127 ≈ 0.00787; stored integers 64, −32, 127; restored ≈ 0.504, −0.252, 1.000.

## Appendix E — Where to Go Next, and Sources

**Build intuition**

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) — animated, visual introduction to vectors and matrices.
- [3Blue1Brown: Neural Networks](https://www.3blue1brown.com/topics/neural-networks) — gradient descent, backpropagation and transformers, visually.
- [Jay Alammar: The Illustrated Transformer](https://jalammar.github.io/illustrated-transformer/) — the classic picture-by-picture walkthrough of attention.

**Build it yourself**

- [Andrej Karpathy: Neural Networks: Zero to Hero](https://karpathy.ai/zero-to-hero.html) — from a hand-written backprop engine to a small GPT, in code.
- [Dive into Deep Learning](https://d2l.ai) — a free, interactive textbook with runnable code.
- [Stanford CS224N: Natural Language Processing with Deep Learning](https://web.stanford.edu/class/cs224n/) — university course materials.

**Original papers mentioned in this book**

- [Vaswani et al., 2017. Attention Is All You Need](https://arxiv.org/abs/1706.03762) — the transformer (Chapter 14).
- [Su et al., 2021. RoFormer: Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE (Chapter 15).
- [Kaplan et al., 2020. Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — scaling laws (Chapter 16).
- [Hoffmann et al., 2022. Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — the "Chinchilla" result (Chapter 16).
- [Ouyang et al., 2022. Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — RLHF for assistants (Chapter 16).
- [Rafailov et al., 2023. Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO (Chapter 16).
- [Touvron et al., 2023. Llama 2](https://arxiv.org/abs/2307.09288) — model sizes cited in Chapter 15.
- [Elhage et al., 2022. Toy Models of Superposition](https://transformer-circuits.pub/2022/toy_model/index.html) — superposition (Chapter 13).
- [Olsson et al., 2022. In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — induction heads (Chapter 14).

**A note on the numbers in this book.** Figures labelled "illustrative" use made-up but realistic values chosen to teach a pattern; they are not measurements from a specific model. Worked examples are computed exactly from the numbers given. Model sizes and rules of thumb are approximate and drawn from the published papers above.
