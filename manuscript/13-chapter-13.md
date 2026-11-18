## Chapter 13 — Neural Networks from Scratch

A neural network is a stack of layers, and each layer does three things: multiply by a weight matrix, add a bias, and apply a simple bending function called an activation. You already know the first two from Part II. This chapter adds the third, builds a tiny network you can compute by hand, trains it one step, and shows where these networks live inside an LLM.

### 13.1 One neuron: a tiny decision maker

Should you go to the beach today? You might weigh a few factors: is it sunny, is it warm, are your friends free? Some factors matter more than others, and you need enough "yes" overall to bother going.

An artificial **neuron** works exactly like this:

1. Take the inputs (each a number).
2. Multiply each by its own **weight** (how much it matters).
3. Add them up and add a **bias** (how eager the neuron is to say yes in general).
4. Pass the total through an **activation function** that turns it into the output.

![Figure 13.1 · One neuron with three inputs, made-up weights](figures/fig-13-01.png)

In symbols, a neuron computes:

$$
\text{output} = f\left(\sum_i w_i x_i + b\right) = f(\mathbf{w} \cdot \mathbf{x} + b)
$$

The part inside the parentheses is the weighted sum from Chapter 2, which is the dot product from Chapter 4, plus a bias. The neuron is literally measuring how well the input agrees with its weight vector, then deciding how strongly to respond.

The name comes from a loose analogy with brain cells, which receive signals from other cells and "fire" when the combined input is strong enough. Artificial neurons are vastly simpler than real ones; the analogy is inspiration, not biology.

### 13.2 Activation functions: the essential bend

Why not skip the activation and just output the weighted sum? Because then the network could only ever compute straight-line relationships, and the world is not straight.

Here are the most common activation functions:

![Figure 13.2 · ReLU, GELU (approximation) and sigmoid, computed](figures/fig-13-02.png)

| Function | Rule | Shape | Where it is used |
| --- | --- | --- | --- |
| Sigmoid | 1 / (1 + e⁻ᶻ) | S-curve squashing everything into 0 to 1 | Older networks; "on/off" gates |
| ReLU | max(0, z) | Zero for negatives, straight line for positives | Very common, simple and fast |
| GELU | z × (a smooth version of "is z positive?") | Like ReLU, but with a smooth corner | Many transformers |
| SwiGLU | A gated variant combining two projections | Like GELU, with an extra learned gate | Many recent LLMs |

ReLU (rectified linear unit) is the easiest to understand: **pass positive values through unchanged, block negative values to zero.** Think of a one-way valve on a pipe: water flows one direction only. That single kink is enough to give networks their power.

### 13.3 Why the bend matters

Here is a surprising fact. Stack two layers *without* activation functions:

$$
W_2 (W_1 \mathbf{x}) = (W_2 W_1) \mathbf{x}
$$

The product W₂W₁ is just another matrix. So two linear layers are no more powerful than one. A hundred linear layers are no more powerful than one. All that depth is wasted.

Add an activation between them and everything changes. Think of folding a sheet of paper. One fold makes a single crease. Fold, then fold again in another direction, then again, and you can make complex shapes, like origami. Each layer "folds" the space with its activation; many layers together can bend it into almost any shape.

Mathematicians have proven a remarkable result, the **universal approximation theorem**: a network with enough neurons and a non-linear activation can approximate any reasonable function as closely as you like. It does not say how to find the weights (that is gradient descent's job) or how many neurons you need, but it says the capacity is there.

> **Real-world analogy.** Pricing a taxi ride is linear: a start fee plus a rate per km. But real decisions have thresholds and interactions: "I'll go to the beach only if it's sunny **and** warm", or "a discount applies only above $100". Lines cannot express "only if". Bends can.

### 13.4 Layers: many neurons at once

One neuron makes one decision. A **layer** is many neurons looking at the same inputs, each with its own weights, each detecting something different. Stack the weight vectors of all the neurons as the rows of a matrix, and the whole layer is a single matrix multiplication (Chapter 5):

$$
\mathbf{h} = f(W \mathbf{x} + \mathbf{b})
$$

![Figure 13.3 · A tiny fully connected network, 3 → 4 → 2](figures/fig-13-03.png)

The layers between input and output are called **hidden layers**, because you never see their values directly; they are the network's private working notes. Every input connects to every neuron in the next layer, so such a layer is called **fully connected** or **dense**.

### 13.5 A tiny network, computed by hand

Let's compute a network with 2 inputs, 2 hidden neurons (with ReLU) and 1 output. Here are its weights:

| Layer | Weights | Biases |
| --- | --- | --- |
| Hidden | W₁ = \[\[0.5, −1\], \[1, 0.5\]\] | b₁ = \[0, −1\] |
| Output | W₂ = \[2, 3\] | b₂ = 0.5 |

Feed in the input x = \[1, 2\].

**Step 1: hidden layer weighted sums.**

- Neuron 1: 0.5 × 1 + (−1) × 2 + 0 = −1.5
- Neuron 2: 1 × 1 + 0.5 × 2 + (−1) = 1

**Step 2: ReLU.** Negative values become zero: h = \[0, 1\]. Neuron 1 is "silent" for this input; neuron 2 is active.

**Step 3: output.** y = 2 × 0 + 3 × 1 + 0.5 = **3.5**.

That is a complete forward pass. A real LLM does the same thing with vectors of thousands of numbers and dozens of layers, but there is no step you have not just done by hand.

### 13.6 Training the tiny network one step

Suppose the correct answer for this input is 3. Using squared error, the loss is (3.5 − 3)² = 0.25. Now apply backpropagation (Chapter 11) and gradient descent (Chapter 12) with learning rate 0.1.

1. **Loss to output.** L = (y − 3)², so ∂L/∂y = 2(y − 3) = 1.
2. **Output weights.** y = W₂ · h + b₂, so ∂y/∂W₂ = h = \[0, 1\] and ∂y/∂b₂ = 1. By the chain rule, ∂L/∂W₂ = 1 × \[0, 1\] = \[0, 1\] and ∂L/∂b₂ = 1.
3. **Update.** W₂ becomes \[2 − 0.1 × 0, 3 − 0.1 × 1\] = \[2, 2.9\]. b₂ becomes 0.5 − 0.1 = 0.4.
4. **Check.** New output: 2 × 0 + 2.9 × 1 + 0.4 = 3.3. New loss: (3.3 − 3)² = 0.09, down from 0.25.

The hidden-layer weights get gradients too, by continuing the chain rule backward through ReLU; we skip that arithmetic here. But notice one thing: hidden neuron 1 output zero, so its weight in W₂ got zero gradient and did not change. A neuron that is silent for an input does not learn from that input. With ReLU, a neuron that becomes silent for *every* input stops learning entirely, a known problem called a "dead" neuron, and one reason smoother activations like GELU are popular.

> **Developer note.** The same network in PyTorch, with autograd doing step 1 to 3:
>
> ```python
> net = torch.nn.Sequential(torch.nn.Linear(2, 2), torch.nn.ReLU(), torch.nn.Linear(2, 1))
> opt = torch.optim.SGD(net.parameters(), lr=0.1)
> loss = (net(torch.tensor([1., 2.])) - 3) ** 2
> opt.zero_grad(); loss.backward(); opt.step()
> ```

### 13.7 What do neurons learn?

Nobody programs what each neuron detects. Training discovers useful detectors on its own. In image networks, researchers found early-layer neurons detecting edges and colours, middle layers detecting textures and shapes like eyes or wheels, and late layers detecting whole objects like faces or cars. Simple features get combined into complex ones, layer by layer.

Language models show the same pattern. Researchers have found internal features that respond to things like:

- Text inside quotation marks.
- Code written in a particular programming language.
- Mentions of specific places, such as a particular city.
- The sentiment of a sentence, positive or negative.

Often a single neuron responds to several unrelated things, and a single concept is spread across many neurons. It seems networks pack more features than they have neurons by using combinations, a phenomenon researchers call **superposition**. Understanding these internal features is an active field called **interpretability**.

### 13.8 The feed-forward network inside a transformer

Every transformer layer contains a small neural network exactly like the ones in this chapter, called the **feed-forward network** (FFN) or MLP (multi-layer perceptron). It processes each token's vector on its own:

1. **Expand:** multiply by a matrix that turns the d-number vector into about 4d numbers (for d = 4,096, about 16,000; some models use around 11,000 with a gated design).
2. **Activate:** apply GELU or a similar function to every number.
3. **Compress:** multiply by a matrix that brings it back to d numbers.

$$
\text{FFN}(\mathbf{x}) = W_{\text{down}} \; \text{GELU}(W_{\text{up}} \mathbf{x} + \mathbf{b}_1) + \mathbf{b}_2
$$

A useful intuition: attention (Chapter 14) is where tokens **talk to each other**; the feed-forward network is where each token **thinks on its own** about what it has gathered. Many researchers believe much of a model's stored factual knowledge ("Ottawa is the capital of Canada") lives in these feed-forward weights, which hold roughly two-thirds of a transformer's parameters.

### Chapter summary

- A neuron computes a weighted sum plus bias, then applies an activation: f(w · x + b).
- Activation functions (ReLU, GELU, sigmoid) add the bends that let networks model non-linear patterns.
- Without activations, stacked layers collapse into one matrix.
- A layer is many neurons at once: h = f(Wx + b), one matrix multiplication.
- Training uses backprop and gradient descent; silent ReLU neurons get no gradient.
- Each transformer layer contains a feed-forward network that expands, activates and compresses each token's vector.

> **Try it yourself.**
>
> 1. Recompute the tiny network's output for the input x = \[2, 0\].
> 2. A neuron has weights \[1, −2\], bias 0.5 and ReLU. What is its output for \[3, 1\]? For \[1, 1\]?
> 3. A feed-forward layer has d = 1,024 and expands to 4,096. How many weights are in its two matrices together?
