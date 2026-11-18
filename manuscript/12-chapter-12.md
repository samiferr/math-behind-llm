## Chapter 12 — Gradient Descent: How Models Learn

Gradient descent is the learning algorithm: compute the gradient of the loss, take a small step in the opposite direction, and repeat. Every LLM you have used was trained this way, with refinements. This chapter shows the basic rule, the one setting that matters most (the learning rate), the tricks that make it work at scale, and the trap of overfitting.

### 12.1 Walking downhill in fog

Imagine you are on a mountainside in thick fog and want to reach the valley floor. You cannot see the valley. You can only feel the ground under your feet. A sensible plan:

1. Feel which direction slopes downward most steeply.
2. Take a step that way.
3. Repeat until the ground feels flat.

That is gradient descent. The "ground under your feet" is the gradient (Chapter 11), which you can compute at your current position. You never need a map of the whole landscape, which is fortunate, because for an LLM the landscape has billions of dimensions.

### 12.2 The update rule

In symbols, one step of gradient descent is:

$$
w_{\text{new}} = w_{\text{old}} - \eta \cdot \frac{\partial L}{\partial w}
$$

The Greek letter η (eta) is the **learning rate**: how big a step to take. The minus sign is what makes us go downhill. If the gradient is negative (the loss falls as w increases), subtracting a negative number increases w. If the gradient is positive, w decreases. Either way, we move toward lower loss.

**Worked example.** Use the loss from Chapter 10, L(w) = (w − 3)², whose gradient is 2(w − 3). Start at w = 0 with learning rate η = 0.25.

| Step | w | Gradient 2(w − 3) | Update: w − 0.25 × gradient | Loss |
| --- | --- | --- | --- | --- |
| 0 | 0 | −6 | 0 + 1.5 = 1.5 | 9 |
| 1 | 1.5 | −3 | 1.5 + 0.75 = 2.25 | 2.25 |
| 2 | 2.25 | −1.5 | 2.25 + 0.375 = 2.625 | 0.56 |
| 3 | 2.625 | −0.75 | 2.625 + 0.1875 = 2.81 | 0.14 |
| 4 | 2.81 | −0.375 | 2.91 | 0.035 |

![Figure 12.1 · Five steps computed from the update rule](figures/fig-12-01.png)

Two things to notice. First, the model homes in on w = 3 without ever being told the answer. Second, the steps get smaller automatically. Near the bottom the slope is gentle, so the gradient is small, so each step is small. Gradient descent naturally slows down as it arrives.

### 12.3 The learning rate: the most important dial

The learning rate is like the stride length of our hiker in the fog.

![Figure 12.2 · Loss computed for each learning rate; the red line leaves the chart after step 3](figures/fig-12-02.png)

| Learning rate | What happens | Hiker analogy |
| --- | --- | --- |
| Too small | Loss falls painfully slowly; training takes forever | Shuffling forward in tiny steps |
| About right | Loss falls quickly and settles | Confident, sensible strides |
| Too large | Steps overshoot the valley, bounce from side to side, and can fly off entirely | Leaping so far you land higher up the opposite slope |

For real LLMs, learning rates are often tiny numbers like 0.0003. Choosing it is part science, part experience, and researchers spend real effort tuning it, because a bad choice can waste millions of dollars of computing time.

> **Common confusion.** "A bigger learning rate means faster learning." Only up to a point. Past a certain size, larger steps make learning slower or impossible, as the red line in Figure 12.2 shows.

### 12.4 All the weights at once

An LLM does not have one weight; it has billions. The update rule works exactly the same, applied to every weight simultaneously. Written with vectors:

$$
\mathbf{w}_{\text{new}} = \mathbf{w}_{\text{old}} - \eta \, \nabla L
$$

The gradient ∇L is a vector with one entry per weight, computed by backpropagation. One step of training updates every weight in the model at the same time, each by its own amount.

### 12.5 Mini-batches: learning from samples

The "true" loss of an LLM is its average loss over all its training text: trillions of tokens. Computing the exact gradient of that would mean reading the entire dataset before taking a single step. Far too slow.

Instead, each step uses a small random sample of text called a **mini-batch** (perhaps a few million tokens for a large model) and computes the gradient on that sample alone. This is called **stochastic gradient descent** (SGD); "stochastic" means "involving randomness".

Think of a political poll. To estimate how a whole country will vote, pollsters do not ask every citizen; they ask a well-chosen sample of a thousand or so. The answer is a bit noisy, but it points in the right direction and is vastly cheaper. Each mini-batch gradient is a poll of the full gradient.

The noise is not all bad. The slightly random steps can jostle the model out of poor spots in the landscape, a little like shaking a jar of mixed nuts helps them settle better.

| Term | Meaning |
| --- | --- |
| Batch size | How many examples (or tokens) are used for one gradient step |
| Step (or iteration) | One gradient computation and one update of all weights |
| Epoch | One full pass through the training data. Large LLMs often see most of their data only about once |

### 12.6 Momentum and Adam: smarter steps

Plain gradient descent has two weaknesses. In long, narrow valleys it zigzags from wall to wall instead of running down the middle. And one learning rate for all weights is a compromise: some weights need big steps, others small ones.

**Momentum** fixes the zigzag. Instead of following only the current gradient, keep a running average of recent gradients, like a heavy ball rolling downhill. A ball does not stop and turn instantly at every bump; it builds up speed in consistent directions and smooths out back-and-forth wobbles.

**Adam** (short for "adaptive moment estimation") adds a second idea: give each weight its own effective step size. It tracks how large each weight's gradients have recently been. Weights with consistently large, noisy gradients get smaller steps; weights with small, rare gradients get relatively larger ones. It is like a teacher who gives quieter students more encouragement and asks louder ones to slow down.

Adam, or a close variant called AdamW, is the standard optimizer for training LLMs. You do not need its formulas to understand LLMs; just remember that it is gradient descent with momentum and per-weight step sizes.

### 12.7 Learning rate schedules

The best learning rate changes during training. A common recipe for LLMs:

1. **Warm-up.** Start with a tiny learning rate and raise it gradually over the first few thousand steps. At the start the weights are random and gradients are wild; big steps would be reckless.
2. **Cruise.** Train at the full learning rate for most of the run.
3. **Decay.** Slowly lower the learning rate toward the end, so the model can settle precisely into a good spot.

It is exactly how you drive: ease out of the driveway, cruise on the highway, slow down to park.

### 12.8 Local minima and saddle points

A natural worry: what if gradient descent gets stuck in a small dip that is not the lowest point, a **local minimum**? In two dimensions this is a real risk, like a hiker trapped in a small hollow halfway down the mountain.

In billions of dimensions it turns out to be much less of a problem. For a point to be a true trap, the ground must curve upward in *every one* of the billions of directions at once. That is rare. Far more common are **saddle points**, shaped like a horse's saddle or a mountain pass: downhill in some directions, uphill in others. Momentum and the noise of mini-batches help the model roll off saddles. In practice, large models trained with gradient descent reliably reach good solutions.

### 12.9 Overfitting: memorizing instead of learning

A student who memorizes last year's exam answers can score 100% on last year's exam and fail this year's. A model can do the same thing: become excellent on the exact text it trained on but poor on new text. This is **overfitting**.

To detect it, hold back some data the model never trains on, called the **validation set**, and watch the loss on it:

![Figure 12.3 · Illustrative training run](figures/fig-12-03.png)

While both curves fall, the model is learning general patterns. When the training loss keeps falling but the validation loss turns upward, the model has started memorizing quirks of its training data. The best model is the one at the bottom of the validation curve.

Ways to fight overfitting:

- **More data.** The surest cure. This is one reason LLMs train on such enormous datasets: with trillions of tokens, there is too much to memorize, so the model must learn general patterns.
- **Weight decay.** Gently shrink all weights toward zero at each step, which discourages extreme, overly specific solutions.
- **Dropout.** Randomly switch off some neurons during training, so the network cannot rely on any single path. Like a team that practises with random players benched, so everyone learns to contribute.
- **Early stopping.** Stop training when the validation loss stops improving.

### 12.10 The complete training loop

You now know every piece of how a model learns. Here is the whole loop:

1. Initialize all weights with small random numbers.
2. Take a mini-batch of training text.
3. **Forward pass:** run the model and compute the loss (Chapter 9).
4. **Backward pass:** compute the gradient of the loss for every weight with backpropagation (Chapter 11).
5. **Update:** nudge every weight against its gradient using the optimizer (this chapter).
6. Repeat from step 2, hundreds of thousands of times, checking the validation loss now and then.

> **Developer note.** In PyTorch the loop is a few lines:
>
> ```python
> opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.1)
> for batch in loader:
>     logits = model(batch.inputs)
>     loss = F.cross_entropy(logits.view(-1, V), batch.targets.view(-1))
>     opt.zero_grad(); loss.backward(); opt.step()
> ```

### Chapter summary

- Gradient descent repeatedly steps against the gradient: w ← w − η ∇L.
- The learning rate η controls step size: too small is slow, too large diverges.
- Mini-batches estimate the gradient from a sample, making each step cheap.
- Momentum smooths steps; Adam gives each weight its own step size.
- Learning rate schedules warm up, cruise and decay.
- Overfitting is memorization; a validation set detects it, and more data, weight decay and dropout reduce it.

> **Try it yourself.**
>
> 1. Redo the worked example in 12.2 with η = 0.5. What happens after one step? Why?
> 2. Redo it with η = 1.0. Describe what w does over three steps.
> 3. Your training loss is 1.2 and still falling, but validation loss rose from 2.0 to 2.3 over the last epochs. What is happening and what would you do?
