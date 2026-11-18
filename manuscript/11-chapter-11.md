## Chapter 11 — Gradients and the Chain Rule

The gradient is a list of derivatives, one per input, and it points in the direction that increases a function fastest. The chain rule tells you how to compute derivatives through a long chain of functions by multiplying simple local slopes. Together they make **backpropagation**, the algorithm that computes billions of derivatives for an LLM in about the same time it takes to run the model a couple of times.

### 11.1 Functions of many inputs: one knob at a time

A pizza's price depends on its size and its number of toppings. A recording studio's mixing desk has dozens of sliders, each changing the sound. An LLM's loss depends on billions of weights. How do you take a derivative when there are many inputs?

The answer: **wiggle one input at a time, holding all the others fixed.** The result is called a **partial derivative**, written with a curly ∂ instead of d.

Take a simple function of two inputs:

$$
f(x, y) = x^2 + 3y
$$

- Holding y fixed and wiggling x: the 3y part does not change, so it acts like a constant. The derivative of x² is 2x. So ∂f/∂x = 2x.
- Holding x fixed and wiggling y: the x² part does not change. The derivative of 3y is 3. So ∂f/∂y = 3.

At the point (x, y) = (2, 5): ∂f/∂x = 4 and ∂f/∂y = 3. A small nudge to x matters a bit more than the same nudge to y.

> **Real-world example.** A coffee shop's daily profit depends on the price of a latte and on the hours it stays open. The owner can ask two separate questions: "if I raise the price by 10 cents, keeping hours the same, what happens?" and "if I stay open one more hour, keeping the price the same, what happens?" Those are two partial derivatives.

### 11.2 The gradient: all the partial derivatives in one vector

Collect all the partial derivatives into a vector and you have the **gradient**, written ∇f (the symbol is called "nabla" or "del"):

$$
\nabla f = \left[ \frac{\partial f}{\partial x}, \; \frac{\partial f}{\partial y} \right]
$$

At (2, 5) our gradient is \[4, 3\]. Because it is a vector, it has a direction and a length, and both mean something:

- **Direction:** the gradient points in the direction of **steepest increase**. If you are standing on a hillside, the gradient points straight uphill.
- **Length:** how steep that uphill direction is. A long gradient means a steep slope; a gradient of length zero means flat ground.

For training we want to go *down* the loss, so we step in the opposite direction: **minus the gradient**.

![Figure 11.1 · Schematic loss landscape for two weights](figures/fig-11-01.png)

Figure 11.1 is drawn like a hiking map with contour lines. Each ring connects points of equal loss. The gradient at your position is always perpendicular to the ring you are standing on, because that is the steepest way to change height.

For an LLM, the gradient has one entry per parameter, billions of entries. You cannot draw a billion-dimensional hillside, but the meaning is identical: a giant vector saying, for each weight, which way to nudge it to raise the loss fastest. Flip its sign and you know how to lower the loss fastest.

### 11.3 The chain rule: gears and exchange rates

An LLM's loss does not depend on a weight directly. A weight in layer 1 affects layer 1's output, which affects layer 2, which affects layer 3, and so on until the loss. It is a chain of functions (Chapter 2, composition). How do you take a derivative through a chain?

Think of interlocking gears. Gear A turns gear B 3 times for each of its turns. Gear B turns gear C 2 times for each of its turns. How many times does C turn for each turn of A? **3 × 2 = 6.** Rates of change through a chain **multiply**.

Or think of currencies. 1 Canadian dollar buys 0.72 US dollars. 1 US dollar buys 0.93 euros. So 1 Canadian dollar buys 0.72 × 0.93 ≈ 0.67 euros. The conversion rates multiply.

That is the **chain rule**:

$$
\frac{dy}{dx} = \frac{dy}{du} \cdot \frac{du}{dx}
$$

If y depends on u, and u depends on x, the sensitivity of y to x is the sensitivity of y to u times the sensitivity of u to x. For longer chains, keep multiplying:

$$
\frac{dL}{dw} = \frac{dL}{dh_3} \cdot \frac{dh_3}{dh_2} \cdot \frac{dh_2}{dh_1} \cdot \frac{dh_1}{dw}
$$

Each factor is a **local** derivative: how one step reacts to its immediate input. Each is easy on its own. The chain rule glues them together.

> **Try it yourself.** If y = (3x + 1)², let u = 3x + 1 so y = u². Then dy/du = 2u and du/dx = 3. So dy/dx = 2u × 3 = 6(3x + 1). At x = 1, that is 24.

### 11.4 A worked example: training one weight

Let's make this concrete with the smallest possible "model": a single weight w that multiplies an input x to make a prediction. The loss is the squared error between the prediction and the target y:

$$
L = (w \cdot x - y)^2
$$

With x = 2, target y = 10 and current weight w = 3, break the computation into simple steps and run it **forward**:

1. Prediction: z = w × x = 6
2. Error: e = z − y = −4
3. Loss: L = e² = 16

Now go **backward**, finding each step's local slope:

1. How does L change with e? L = e², so ∂L/∂e = 2e = −8.
2. How does e change with z? e = z − 10, so ∂e/∂z = 1.
3. How does z change with w? z = w × 2, so ∂z/∂w = 2.

Multiply them, following the chain rule:

![Figure 11.2 · A three-step computation, forward and backward](figures/fig-11-02.png)

The gradient ∂L/∂w = −16 says: increasing w will decrease the loss, and quite strongly. Sure enough, the perfect weight is w = 5 (since 5 × 2 = 10). The derivative pointed the right way without knowing the answer.

### 11.5 Backpropagation: blame flows backward

What you just did by hand is **backpropagation** ("backprop"): run the computation forward, then walk backward from the loss, multiplying local slopes, to find how much each earlier quantity contributed.

A real-world analogy: a restaurant gets a complaint that a dish was too salty. The manager traces it backward. The dish came from the cook, who used a sauce, which was made by the prep station, which used a new brand of stock. At each step the manager asks a local question, "how much did this step contribute to the saltiness of the next?", and multiplies the answers to find how much each person or ingredient is to blame. Backprop is this blame assignment, done with exact numbers.

Three facts make backprop the engine of deep learning:

1. **It reuses work.** The slope ∂L/∂e = −8 was computed once and used for everything earlier in the chain. In a network, the backward signal from layer 80 is reused for every weight in layer 79, which is reused for layer 78, and so on. Nothing is computed twice.
2. **It is cheap.** Computing the gradient for all parameters costs only about two to three times as much as running the model forward once, whether the model has ten weights or a trillion.
3. **It is automatic.** Each building block (multiply, add, softmax, and so on) only needs to know its own local slope. Software libraries combine them automatically. This is called **automatic differentiation**.

> **Developer note.** PyTorch does backprop for you. Here is the example from 11.4:
>
> ```python
> import torch
> w = torch.tensor(3.0, requires_grad=True)
> x, y = 2.0, 10.0
> loss = (w * x - y) ** 2
> loss.backward()
> w.grad   # tensor(-16.)
> ```

### 11.6 When gradients vanish or explode

The chain rule multiplies many numbers together. That can cause trouble in deep networks with dozens of layers:

| Each local slope is about | After 50 layers | Effect |
| --- | --- | --- |
| 0.5 | 0.5⁵⁰ ≈ 0.000000000000001 | **Vanishing gradient**: early layers get almost no signal and stop learning |
| 1.0 | 1.0⁵⁰ = 1 | Healthy: signal passes through intact |
| 1.5 | 1.5⁵⁰ ≈ 637,000,000 | **Exploding gradient**: updates become huge and training blows up |

This is like a game of telephone over 50 people: a message that loses a little at each step is unrecognizable at the end, and one that gets exaggerated at each step becomes absurd. Deep networks must keep local slopes close to 1.

Transformers solve this with two tricks you will meet in Chapter 15. **Residual connections** add each layer's output to its input, creating a "highway" along which the gradient can flow backward without being multiplied down. **Layer normalization** keeps the numbers flowing through the network at a steady scale. Without these, models with 80 or more layers could not be trained.

### Chapter summary

- A partial derivative wiggles one input while holding the others fixed.
- The gradient collects all partial derivatives into a vector pointing steepest uphill; minus the gradient points steepest downhill.
- The chain rule says sensitivities through a chain multiply, like gear ratios or exchange rates.
- Backpropagation runs the model forward, then multiplies local slopes backward to get every parameter's gradient efficiently.
- Multiplying many slopes can make gradients vanish or explode; residual connections and normalization prevent this.

> **Try it yourself.**
>
> 1. For f(x, y) = 2x + y², find the gradient at (1, 3).
> 2. In the 11.4 example, recompute the gradient when w = 5. What does it tell you?
> 3. If three layers have local slopes 0.9, 1.1 and 0.5, what is the overall slope through the chain?
