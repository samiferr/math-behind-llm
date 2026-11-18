## Chapter 2 — Numbers, Functions and Graphs

A function is a rule that turns an input into an output, and an LLM is one giant function: text in, probabilities out. This chapter gives you the five tools you will use on every later page: functions, graphs, slopes, exponentials and logarithms.

### 2.1 A function is a machine

Think of a vending machine. You press B4 and a bag of chips comes out. Press B4 again and you get the same chips. The machine follows a fixed rule: each button maps to one product.

That is all a **function** is: a rule that takes an input and always produces the same output for that input.

Everyday functions are everywhere:

- A recipe: ingredients in, cake out.
- A currency converter: Canadian dollars in, euros out.
- A thermostat: room temperature in, "heat on" or "heat off" out.
- A spell checker: a word in, "correct" or "misspelled" out.

Mathematicians write functions with a short name, usually *f*, and put the input in parentheses:

$$
f(x) = 2x + 1
$$

Read this aloud as "f of x equals two x plus one". It means: take the input, call it *x*, multiply it by 2, then add 1. So:

- f(3) = 2 × 3 + 1 = 7
- f(10) = 2 × 10 + 1 = 21
- f(0) = 2 × 0 + 1 = 1

The letter *x* is just a placeholder, a name for "whatever you put in". You could equally write f(apple) = 2 × apple + 1. Mathematicians use *x* out of habit.

> **LLM connection.** An LLM is a function. Its input is a sequence of tokens. Its output is a list of probabilities, one for every token in the vocabulary. Inside, it is built from thousands of small functions chained together. If you understand small functions, you understand the parts.

### 2.2 Tables of inputs and outputs

One way to understand a function is to try several inputs and write down the results. Here is the function that converts Celsius to Fahrenheit, which every Canadian travelling to the United States eventually needs:

$$
F(c) = 1.8c + 32
$$

| Celsius (input) | Calculation | Fahrenheit (output) |
| --- | --- | --- |
| −20 | 1.8 × (−20) + 32 | −4 |
| 0 | 1.8 × 0 + 32 | 32 |
| 20 | 1.8 × 20 + 32 | 68 |
| 37 | 1.8 × 37 + 32 | 98.6 |
| 100 | 1.8 × 100 + 32 | 212 |

A table is useful, but it only shows a few points. A **graph** shows all of them at once.

### 2.3 Drawing a function: the graph

A graph is a picture of a function. You draw two lines at right angles:

- The horizontal line, the **x-axis**, holds the inputs.
- The vertical line, the **y-axis**, holds the outputs.

For each input you put a dot at (input, output). Join enough dots and you see the function's shape. The pair (input, output) is called a **point** and is written in parentheses: the point (20, 68) means "input 20, output 68".

The great power of a graph is that your eye reads its shape instantly. Is it going up or down? Is it straight or curved? Does it flatten out? Those questions, answered at a glance, are often more useful than any single number.

### 2.4 Straight lines: slope and intercept

Let's take a taxi. In this city the fare is a $3.50 start fee plus $2 for every kilometre.

$$
\text{fare}(k) = 3.50 + 2k
$$

![Figure 2.1 · Taxi fare example, 0 to 10 km](figures/fig-02-01.png)

Two numbers completely describe this line:

- **The intercept** ($3.50) is where the line crosses the vertical axis: the output when the input is zero. You pay it before the car moves.
- **The slope** ($2 per km) is how much the output rises for each one-step increase in the input. Look at the dashed step in Figure 2.1: one kilometre across, two dollars up.

Every straight line can be written in this form:

$$
y = mx + b
$$

where *m* is the slope and *b* is the intercept. A bigger slope means a steeper line. A slope of zero means a flat line: the output does not depend on the input at all. A negative slope means the line goes down: for example, the value of a car falling by $3,000 every year.

> **Try it yourself.** A gym charges a $50 joining fee and $30 per month. Write the function for the total cost after *m* months. What is the slope? The intercept? What does the line look like compared with the taxi?

The slope deserves special attention because it returns, in disguise, throughout this book. The slope is a **rate of change**: how much the output changes when the input changes by one. In Chapter 10, derivatives will extend this idea to curved lines, and that extension is the heart of how models learn.

> **LLM connection.** In neural networks, the slope and intercept get new names: the slope is called a **weight** and the intercept is called a **bias**. Each tiny unit inside an LLM computes something very like *weight × input + bias*. The model has billions of these weights and biases, and training is the process of finding good values for them.

### 2.5 Functions with many inputs

Real decisions rarely depend on one thing. A house price depends on its size, its number of bedrooms, its age and its distance from downtown. Here is a made-up pricing rule for a neighbourhood:

$$
\text{price} = 80{,}000 + 250 \times \text{sqft} + 15{,}000 \times \text{bedrooms} - 1{,}000 \times \text{age}
$$

For a 1,600 sq ft, 3-bedroom, 20-year-old house:

| Part | Calculation | Contribution |
| --- | --- | --- |
| Base | 80,000 | $80,000 |
| Size | 250 × 1,600 | $400,000 |
| Bedrooms | 15,000 × 3 | $45,000 |
| Age | −1,000 × 20 | −$20,000 |
| **Total** |  | **$505,000** |

Notice the pattern. Each input is multiplied by its own number (its weight), the results are added up, and a base amount (the bias) is added. This is called a **weighted sum**, and it is the single most common calculation in all of machine learning. The weights say how much each input matters. A large weight means an input matters a lot. A negative weight means more of that input pushes the output down.

In Chapter 4 you will see that a weighted sum is the same thing as a "dot product" of two vectors. In Chapter 13 you will see that one artificial neuron is exactly a weighted sum followed by one small extra step.

### 2.6 Chaining functions together

Suppose you buy something online from Europe. First the price in euros is converted to Canadian dollars, then 13% Ontario HST is added:

- Step 1: g(euros) = 1.5 × euros (convert currency)
- Step 2: f(dollars) = 1.13 × dollars (add tax)

The total is f(g(euros)): convert first, then tax. For a €100 item, g(100) = 150, then f(150) = 169.50.

Feeding the output of one function into another is called **composition**. It is how you build complicated behaviour from simple pieces, like an assembly line where each station does one job.

$$
(f \circ g)(x) = f(g(x))
$$

The small circle is read "f after g". The order matters. Taxing first and converting second gives the same answer here by coincidence, but in general f(g(x)) and g(f(x)) are different. Putting on socks and then shoes is not the same as shoes then socks.

> **LLM connection.** An LLM is a very long chain of functions: embed the tokens, then layer 1, then layer 2, and so on through layer 80, then turn scores into probabilities. Training works backward through this chain using a rule called the chain rule (Chapter 11), which is all about composition.

### 2.7 Powers and exponents: when things multiply

The taxi fare grows by **adding** the same amount each step. Some things grow by **multiplying** by the same amount each step.

The classic story: a king offers a reward of one grain of rice on the first square of a chessboard, two on the second, four on the third, doubling each time. It sounds cheap. By square 64 the total is more rice than has ever been grown on Earth.

Repeated multiplication is written with an **exponent**:

$$
2^5 = 2 \times 2 \times 2 \times 2 \times 2 = 32
$$

The small raised number says how many times to multiply. Three rules cover almost everything you will need:

| Rule | Example | Why it is true |
| --- | --- | --- |
| aᵐ × aⁿ = aᵐ⁺ⁿ | 2³ × 2² = 2⁵ | Three 2s times two 2s is five 2s |
| a⁰ = 1 | 7⁰ = 1 | Multiplying by "nothing" leaves 1 |
| a⁻ⁿ = 1 / aⁿ | 2⁻³ = 1/8 | A negative exponent means divide instead |

**The special number e.** One base shows up so often in nature and finance that it has its own letter: e ≈ 2.718. Where does it come from? Put $1 in a bank account paying 100% interest per year:

| How often interest is added | Money after one year |
| --- | --- |
| Once a year | $2.00 |
| Twice a year | $2.25 |
| Monthly | $2.61 |
| Daily | $2.71 |
| Continuously | $2.718… = e |

Compounding more often helps, but the gains level off at e. That is why e is called the base of **natural growth**. The function eˣ is called the **exponential function**, sometimes written exp(x).

![Figure 2.2 · y = eˣ for x from −3 to 3, values rounded](figures/fig-02-02.png)

Look carefully at Figure 2.2. Two properties will matter enormously in Chapter 8:

1. **It is always positive.** Even e⁻³ is a tiny positive number (about 0.05). No input ever gives zero or a negative output.
2. **It exaggerates differences.** Going from x = 1 to x = 2 adds about 4.7. Going from 2 to 3 adds about 12.7. Bigger inputs get much, much bigger outputs.

> **LLM connection.** These two properties are exactly why softmax uses eˣ. A model's raw scores can be negative, but probabilities cannot. Passing every score through eˣ makes them all positive, while making the highest score stand out from the rest.

### 2.8 Logarithms: the undo button

If exponents answer "what do I get if I multiply 2 by itself 5 times?" (32), logarithms answer the reverse question: "how many times must I multiply 2 by itself to get 32?" (5).

$$
\log_2(32) = 5 \quad \text{because} \quad 2^5 = 32
$$

The small 2 is the **base**. With base e we write ln(x), the **natural logarithm**. Real-world scales that use logs include:

- **Earthquakes.** Each step on the magnitude scale is about 32 times more energy. A magnitude 7 is not a little stronger than a 6; it is vastly stronger.
- **Sound.** Decibels are a log scale. 60 dB (conversation) is not twice 30 dB (a whisper); it carries 1,000 times the energy.
- **Doubling time.** "How many years until my savings double?" is a logarithm question.

The one rule that makes logs so useful is this: **logs turn multiplication into addition.**

$$
\ln(a \times b) = \ln(a) + \ln(b)
$$

For example, log₂(8 × 4) = log₂(32) = 5, and log₂(8) + log₂(4) = 3 + 2 = 5.

Now look at what the log does to numbers between 0 and 1, which is where all probabilities live:

![Figure 2.3 · y = ln(p) for p from 0.05 to 1, values rounded](figures/fig-02-03.png)

The log of 1 is 0. The log of anything smaller than 1 is negative, and the smaller the number, the more negative its log. A probability of 0.5 has a log of about −0.69. A probability of 0.05 has a log of about −3.

> **LLM connection.** Logs show up in LLMs for two reasons. First, the probability of a whole sentence is a product of many small probabilities, one per word. Multiplying hundreds of numbers like 0.01 gives a number so small a computer rounds it to zero. Adding their logs instead keeps everything manageable. Second, the **loss** a model minimizes during training is built from the negative log of the probability it gave the correct word (Chapter 9). A confident correct guess (p near 1) costs almost nothing. A confident wrong guess (p near 0) costs a lot.

### 2.9 Adding up many things: the Σ sign

A grocery receipt adds up many items. Writing "item 1 + item 2 + item 3 + … + item 40" gets tiresome, so mathematicians use the Greek capital letter sigma, Σ, which means "add up":

$$
\sum_{i=1}^{n} x_i = x_1 + x_2 + \dots + x_n
$$

Read it as: "add up x-sub-i, for i going from 1 to n". The small *i* is a counter, like the line number on the receipt. For a receipt with prices $4, $7 and $2, n = 3 and the sum is 13.

The weighted sum from section 2.5 can now be written compactly. If wᵢ are the weights and xᵢ are the inputs:

$$
\text{output} = \sum_{i=1}^{n} w_i x_i + b
$$

This one line is the formula for an artificial neuron (before its final step). You will see it again and again.

> **Developer note.** In Python, Σ is just a loop or `sum()`:
>
> ```python
> output = sum(w * x for w, x in zip(weights, inputs)) + bias
> ```

### Chapter summary

| Idea | One-line meaning | Where it shows up in an LLM |
| --- | --- | --- |
| Function | A rule from input to output | The whole model, and every layer |
| Slope and intercept | Rate of change and starting value | Weights and biases |
| Weighted sum | Multiply each input by its weight, then add | Every neuron, every attention score |
| Composition | One function feeding the next | Layers stacked on layers |
| Exponential eˣ | Always positive, exaggerates differences | Softmax |
| Logarithm ln(x) | Undoes eˣ; turns products into sums | Loss, log-probabilities |
| Sigma Σ | Add up a list | Everywhere |

> **Try it yourself.**
>
> 1. If f(x) = 3x − 2 and g(x) = x², compute f(g(2)) and g(f(2)). Are they equal?
> 2. Use a calculator to find e¹, e² and e³. Check that e¹ × e² equals e³.
> 3. Find ln(0.9), ln(0.5) and ln(0.01). Which one would you least like to be the "probability of the right answer"?
