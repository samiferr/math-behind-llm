# Part I — Foundations

## Chapter 1 — What a Language Model Really Does

A large language model (LLM) does one thing: given some text, it predicts what comes next. Everything impressive you have seen an LLM do, from writing code to explaining tax law, is that single trick repeated thousands of times. This chapter shows you the whole machine from the outside before we open it up.

### 1.1 The world's most ambitious autocomplete

Your phone keyboard already does a tiny version of what an LLM does. Type "See you" and it suggests "tomorrow", "soon" or "later". It has learned, from millions of messages, which words tend to follow "See you".

An LLM is the same idea scaled up enormously:

- Your keyboard looks at the last two or three words. An LLM can look at hundreds of thousands of words at once.
- Your keyboard learned from text messages. An LLM learned from a large share of the books, articles, code and conversations ever written down.
- Your keyboard has a few million adjustable numbers inside it. A large LLM has hundreds of billions.

When you ask an LLM a question, it does not look up an answer in a database. It writes the answer one small piece at a time, each time asking itself: *given everything so far, what is the most sensible next piece?*

### 1.2 A game you already play

Try this. Fill in the blank:

> The cat sat on the \_\_\_

You probably thought "mat". Maybe "sofa", "floor" or "windowsill". You almost certainly did not think "democracy" or "seventeen".

Notice three things about what your brain just did:

1. **You did not pick one answer with certainty.** Several words felt possible. Some felt more likely than others.
2. **You used context.** The words "cat" and "sat on" ruled out most of the dictionary.
3. **You used experience.** You have heard the phrase "the cat sat on the mat" before, so "mat" jumped out.

An LLM does exactly these three things, and each one maps to a branch of math:

| What your brain did | What the LLM does | The math behind it |
| --- | --- | --- |
| Felt several answers as more or less likely | Gives every possible next word a probability | Probability (Chapters 7–9) |
| Used the surrounding words | Mixes information between words with "attention" | Linear algebra (Chapters 3–6, 14) |
| Drew on past experience | Adjusts billions of internal numbers while reading training text | Calculus and optimization (Chapters 10–13) |

Here is what that "several answers, some more likely" feeling looks like when a model expresses it with numbers:

![Figure 1.2 · Illustrative probabilities, not from a specific model](figures/fig-01-02.png)

The model does not "know" the answer is "mat". It holds a spread of beliefs, and "mat" happens to get the biggest share. We call this spread a **probability distribution**, and you will learn to build one by hand in Chapter 8.

### 1.3 Why words must become numbers

Computers cannot do anything with the word "cat" directly. A computer chip can only add, multiply and compare numbers. So before a model can reason about text, every piece of text must be converted into numbers.

The naive way is to give each word an ID number: "the" = 1, "cat" = 2, "sat" = 3, and so on. This works for storage, but it hides all meaning. Is "cat" (2) closer in meaning to "sat" (3) than to "dog" (847)? The ID numbers say yes, which is nonsense.

The clever way, which LLMs use, is to describe each word with a long *list* of numbers, where each position in the list captures some aspect of meaning. Words with similar meanings end up with similar lists. Such a list is called a **vector**, and the specific vector a model assigns to a word is called an **embedding**. Vectors are the subject of Chapter 3, and embeddings of Chapter 6.

> **Real-world example.** A real-estate website describes a house with a list: \[3 bedrooms, 2 bathrooms, 1,800 sq ft, 12 km from downtown, built 1995\]. Two houses with similar lists are similar houses. An LLM describes words the same way, except the list has thousands of positions and nobody chose what they mean; the model discovered them.

### 1.4 The pipeline, step by step

Here is the whole journey from your sentence to the next word. Each box is one stage, and the right-hand column shows which math makes that stage possible.

![Figure 1.1 · The LLM pipeline, six steps and one training loop](figures/fig-01-01.png)

Let's walk through it in plain words.

**Step 1 — Text.** You type "The cat sat on the".

**Step 2 — Tokens.** The text is chopped into pieces called **tokens**. Often a token is a whole word, but long or rare words get split: "unbelievable" might become "un", "believ", "able". Each token has an ID number from a fixed vocabulary of perhaps 50,000 to 200,000 pieces.

**Step 3 — Embeddings.** Each token ID is replaced by its vector: a list of, say, 4,096 numbers. Now the sentence is a stack of vectors, one per token.

**Step 4 — Transformer layers.** The vectors pass through dozens of identical layers. In each layer two things happen:

- **Attention** lets each word look at the other words and pull in information from the relevant ones. In "The bank of the river", attention helps "bank" learn it means a riverside, not a building with money.
- **A feed-forward network** processes each word's vector on its own, like a small calculator that refines the meaning.

Both of these are built from matrix multiplication, which you will master in Chapter 5.

**Step 5 — Probabilities.** After the last layer, the model produces one score for every token in its vocabulary. A function called **softmax** turns these scores into probabilities that add up to 100%.

**Step 6 — Next token.** The model picks one token, usually a likely one but not always the single most likely. That token is added to the text, and the whole process runs again to get the next one. This is why chatbot answers appear word by word.

### 1.5 Learning from examples

Where do the billions of numbers inside the model come from? Nobody types them in. They are **learned**.

Imagine teaching a child to throw a ball into a basket:

1. The child throws.
2. The ball lands 30 cm to the left.
3. You say "a little more to the right".
4. The child adjusts and throws again.

After enough throws, the child is accurate. Training an LLM works the same way:

1. The model reads some real text, for example "The cat sat on the mat", but only sees up to "the".
2. It predicts the next word, perhaps giving "mat" only 5% probability.
3. A formula called the **loss** measures how wrong that was. Giving the true answer only 5% is quite wrong.
4. **Calculus** tells the model exactly which direction to nudge each of its billions of numbers so that next time "mat" gets a higher probability.
5. Repeat for trillions of words.

The dashed loop in Figure 1.1 is this learning process. It runs only during training. When you chat with a finished model, the numbers are frozen and only the forward pipeline runs.

> **Common confusion.** "The model learns from my conversations as I chat." Usually not, at least not in the moment. The learning happens beforehand, during training. During a chat the weights stay fixed; the model only uses what is in the conversation as context.

### 1.6 The four kinds of math you need

Everything in this book falls into four families. Here is each in one sentence, with an everyday analogy.

| Math family | One-sentence idea | Everyday analogy | Where it lives in an LLM |
| --- | --- | --- | --- |
| Functions and graphs | A rule that turns inputs into outputs | A coffee machine: beans and water in, coffee out | Every layer is a function |
| Linear algebra | Doing arithmetic on whole lists and tables of numbers at once | A spreadsheet that updates every cell together | Embeddings, attention, every layer's weights |
| Probability | Putting numbers on uncertainty | A weather forecast: 70% chance of rain | Next-token prediction, sampling |
| Calculus | Measuring how a change in one thing changes another | A car's speedometer measures how fast distance changes | Training: deciding how to adjust each weight |

You will meet them in exactly that order. Functions first, because every other idea is a kind of function. Then linear algebra, because it is how a model represents and transforms meaning. Then probability, because it is how a model expresses its guess. Finally calculus, because it is how the model learns.

### 1.7 How big are these models?

A few numbers help make "large" concrete. Exact figures vary by model and many companies do not publish them, so treat these as orders of magnitude:

| Quantity | Typical size | Comparison |
| --- | --- | --- |
| Vocabulary | 50,000–200,000 tokens | A large dictionary |
| Embedding length | 2,000–16,000 numbers per token | A very detailed "profile" of each word |
| Layers | 30–120 | 30–120 rounds of refining meaning |
| Parameters (adjustable numbers) | Billions to trillions | More numbers than there are people on Earth |
| Training text | Trillions of tokens | Millions of books |

None of these numbers needs new math. A model with a trillion parameters uses the same multiplication and addition as a model with ten. That is the good news of this book: once you understand a tiny model, you understand the giant ones.

### Chapter summary

- An LLM predicts the next token, then repeats.
- Its prediction is a probability distribution: many candidates, each with a chance.
- Text becomes tokens, tokens become vectors, vectors flow through transformer layers, and the result becomes probabilities.
- The model's internal numbers are learned by measuring mistakes and nudging the numbers to reduce them.
- Four kinds of math power this: functions, linear algebra, probability and calculus.

> **Try it yourself.** Write down five possible next words for "I'd like a cup of \_\_\_". Give each a percentage so that they add up to 100%. Congratulations: you just built a probability distribution, exactly the kind of object an LLM outputs.
