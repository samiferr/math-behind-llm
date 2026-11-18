# Part IV — Calculus and Learning: How a Model Improves Itself

## Chapter 10 — Derivatives: Measuring Change

A derivative tells you how fast something changes: if you nudge the input a little, how much does the output move? Training an LLM is nothing more than asking this question for every one of its billions of parameters ("if I nudge this weight, does the loss go up or down, and by how much?") and then nudging each one the helpful way. This chapter teaches derivatives with almost no formulas to memorize.

### 10.1 Your car already computes derivatives

A car's odometer shows how far you have gone. Its speedometer shows how fast that distance is changing right now. **Speed is the derivative of distance.**

You already know how to compute an *average* speed: if you drive 120 km in 2 hours, your average speed is 120 / 2 = 60 km/h. That is the slope of a straight line from Chapter 2:

$$
\text{average rate of change} = \frac{\text{change in output}}{\text{change in input}} = \frac{\Delta y}{\Delta x}
$$

The Greek letter Δ (delta) means "change in". But the speedometer does not show your average over the trip. It shows your speed *at this instant*: 0 at a red light, 100 on the highway. How can you have a speed "at an instant", when an instant has no length? That question is what calculus answers.

### 10.2 Curves have a slope at every point

For a straight line, the slope is the same everywhere. For a curve, the slope changes: steep in some places, flat in others. Here is the distance you have travelled on a short drive:

![Figure 10.1 · An illustrative drive, distances rounded](figures/fig-10-01.png)

At the start the curve is nearly flat (you are pulling out of the driveway). In the middle it is steep (you are cruising on a main road). At the end it flattens again (you are braking into a parking spot).

The slope at one point is the slope of the **tangent line**: the straight line that just touches the curve at that point, going the same direction as the curve there. Imagine a tiny ant walking along the curve: the tangent line is the direction the ant is facing at that moment. In Figure 10.1, the tangent at minute 5 has slope 1.2 km per minute, which is 72 km/h. That is what the speedometer showed at minute 5.

**The derivative of a function at a point is the slope of its tangent line there.**

### 10.3 Zooming in

How do you compute the slope at a single point? Zoom in. Any smooth curve, viewed closely enough, looks like a straight line, just as the Earth looks flat when you stand on it.

Let's find the slope of f(x) = x² at x = 1. Take a second point a small step *h* to the right, and compute the average slope between the two:

| Step h | Second point | Average slope = (f(1 + h) − f(1)) / h |
| --- | --- | --- |
| 1 | 2 | (4 − 1) / 1 = 3 |
| 0.5 | 1.5 | (2.25 − 1) / 0.5 = 2.5 |
| 0.1 | 1.1 | (1.21 − 1) / 0.1 = 2.1 |
| 0.01 | 1.01 | (1.0201 − 1) / 0.01 = 2.01 |
| 0.001 | 1.001 | (1.002001 − 1) / 0.001 = 2.001 |

As the step shrinks, the slope closes in on **2**. That target value is the derivative of x² at x = 1. Formally:

$$
f'(x) = \lim_{h \to 0} \frac{f(x + h) - f(x)}{h}
$$

"lim" stands for **limit**: the value the expression approaches as h gets closer and closer to 0. You will never need to compute a limit by hand in this book. The table above is the whole idea.

### 10.4 Notation

You will see the derivative written in several ways. They all mean the same thing.

| Notation | Read as | Who uses it |
| --- | --- | --- |
| f′(x) | "f prime of x" | Short and common |
| dy/dx | "dee y by dee x" | Shows it is a change in y over a change in x |
| df/dx | "dee f by dee x" | Same, for a function named f |
| ∂L/∂w | "partial L by partial w" | Machine learning, for many-input functions (Chapter 11) |

The "d" stands for a tiny change. dy/dx reminds you that the derivative is a ratio of a tiny change in output to a tiny change in input.

### 10.5 A handful of rules

You do not need to zoom in every time. Mathematicians worked out shortcut rules long ago. These cover everything in this book:

| Function | Derivative | Example |
| --- | --- | --- |
| A constant, c | 0 | A flat line has no slope |
| x | 1 | The line y = x rises 1 per step |
| x² | 2x | At x = 1, slope 2 (as we found) |
| xⁿ | n · xⁿ⁻¹ | x³ → 3x² |
| c · f(x) | c · f′(x) | 5x² → 10x |
| f(x) + g(x) | f′(x) + g′(x) | x² + 3x → 2x + 3 |
| eˣ | eˣ | The exponential is its own slope |
| ln(x) | 1 / x | Steep near 0, flat for large x |

Two are worth pausing on. The derivative of eˣ is eˣ itself: the function grows at a rate equal to its own size, like money compounding continuously. And the derivative of ln(x) is 1/x: the log curve in Figure 2.3 is extremely steep for small probabilities, which is why the loss pushes so hard on a model that gave the right answer a tiny probability.

> **Try it yourself.** Using the rules, find the derivative of f(x) = 3x² + 2x + 7. Then evaluate it at x = 2. (Answer: 6x + 2, which is 14 at x = 2.)

### 10.6 The derivative as sensitivity

Here is the interpretation of the derivative that matters most for machine learning:

$$
\text{change in output} \approx f'(x) \times \text{small change in input}
$$

The derivative is a **sensitivity**: how strongly the output reacts to a small nudge of the input. Some real examples:

- **A coffee shop.** At the current price, if the derivative of daily profit with respect to price is −40 dollars per dollar, then raising the price by 10 cents changes profit by about −40 × 0.10 = −$4 a day. Raising prices would hurt.
- **Studying.** If your expected exam grade rises 3 points per extra hour of study near your current level, one more hour is worth about 3 points. The derivative tells you whether more effort is worth it.
- **A thermostat.** If the room warms 0.5°C per minute with the heater on, the controller can predict when to switch it off.

Two pieces of information hide in every derivative:

1. **Its sign** says which direction helps. Positive: increasing the input increases the output. Negative: increasing the input decreases the output.
2. **Its size** says how much it matters. A derivative of 0.001 means the input barely matters right now. A derivative of 50 means it matters a lot.

### 10.7 Finding the bottom of a valley

At the very bottom of a valley, the ground is flat. Mathematically: **at a minimum, the derivative is zero.** And on either side, the sign of the derivative tells you which way is downhill.

![Figure 10.2 · Tangent lines on y = x² at three points](figures/fig-10-02.png)

This is the most important picture in Part IV. Suppose you are standing on the curve and want to reach the bottom:

- At x = −2 the slope is −4: the ground falls as you move right. So move **right**.
- At x = 1.5 the slope is +3: the ground rises as you move right. So move **left**.
- At x = 0 the slope is 0: you are at the bottom. Stop.

Notice the rule: **always move in the direction opposite to the sign of the slope.** That one sentence is the core of how every neural network learns. Chapter 12 turns it into an algorithm called gradient descent.

**A worked example.** Suppose a model has a single weight *w*, and its loss is L(w) = (w − 3)². (In plain words: the loss is smallest when w = 3, but the model does not know that.) The derivative, using the rules and a little algebra, is L′(w) = 2(w − 3).

| Current w | Loss | Slope L′(w) | What the slope says |
| --- | --- | --- | --- |
| 0 | 9 | −6 | Increase w, and strongly |
| 2 | 1 | −2 | Increase w a little |
| 3 | 0 | 0 | Perfect: stop |
| 5 | 4 | +4 | Decrease w |

Without ever being told "the answer is 3", the model can find it by following the slope.

### 10.8 What this means inside an LLM

An LLM's loss (Chapter 9) depends on every one of its parameters. Pick any single weight, say weight number 4,812,337 in layer 17. Hold everything else fixed and ask: if this one weight increased by a hair, would the average loss on this batch of text go up or down, and how much?

The answer is a derivative. If it is negative, increasing that weight helps; if positive, decreasing it helps. Training computes this derivative for every weight, billions of them, on every batch of text, and nudges each one a small step downhill. Repeated millions of times, those tiny nudges turn random numbers into a model that writes poetry and code.

The obvious question is: how do you compute billions of derivatives efficiently, when each weight affects the loss only through a long chain of layers? The answers are the gradient and the chain rule, in Chapter 11.

> **Developer note.** You can always approximate a derivative numerically, exactly as in the zooming table:
>
> ```python
> def derivative(f, x, h=1e-5):
>     return (f(x + h) - f(x - h)) / (2 * h)
> derivative(lambda w: (w - 3) ** 2, 0)   # ≈ -6.0
> ```
>
> This is too slow for billions of weights, but it is the standard way to check that a hand-written derivative is correct.

### Chapter summary

- The derivative is the instantaneous rate of change: the slope of the tangent line.
- It is found by zooming in: the average slope over a smaller and smaller step.
- A few rules (power rule, sums, constants, eˣ, ln x) cover almost everything.
- The derivative is a sensitivity: output change ≈ derivative × input change.
- Its sign tells you which direction increases the output; its size tells you how much.
- At a minimum the derivative is zero; to go downhill, move opposite to the slope's sign.

> **Try it yourself.**
>
> 1. Find the derivative of x³ at x = 2.
> 2. A loss is L(w) = (w + 1)². Its derivative is 2(w + 1). At w = 4, should w increase or decrease?
> 3. Use the zooming method with h = 0.01 to estimate the slope of eˣ at x = 0. Does it match the rule?
