## Preface

You can understand how a large language model works with nothing more than school arithmetic and some patience. This book takes you from "what is a function?" to "how does attention decide which words matter?" one small step at a time. No step assumes something you have not yet seen.

### Who this book is for

This book is for curious people who use tools like ChatGPT or Claude every day and want to know what is happening inside them. It is written for:

- **Developers** who can write code but stopped doing math after school.
- **Students** who want an intuitive foundation before a formal machine learning course.
- **Product managers, analysts and founders** who make decisions about AI and want to reason about it, not just repeat buzzwords.
- **Anyone** who has ever wondered how a computer can "understand" a sentence.

You do not need calculus, linear algebra or statistics. You need to be comfortable with adding, multiplying, fractions and percentages. Everything else is built here.

### How this book teaches

Every idea follows the same five-step rhythm:

1. **A real-world situation** you already understand: shopping, cooking, driving, music, weather.
2. **The intuition**: what the idea is really about, in plain words.
3. **A picture**: a chart or diagram you can take in at a glance.
4. **The math**: the formula, introduced only after you already know what it should say.
5. **The LLM connection**: exactly where this idea lives inside a language model.

You will also find three kinds of boxes throughout the book:

> **Try it yourself** — a small calculation you can do with a pen or a calculator. Doing them is the fastest way to make the ideas stick.

> **Developer note** — a short Python snippet for readers who code. You can skip these without losing the thread.

> **Common confusion** — a mistake almost every beginner makes, and how to avoid it.

### The roadmap

The book has five parts. Each part depends on the ones before it, so read them in order the first time.

| Part | Chapters | What you will learn | Why an LLM needs it |
| --- | --- | --- | --- |
| I. Foundations | 1–2 | What an LLM does; functions, graphs, exponents, logarithms | The vocabulary for everything else |
| II. Linear algebra | 3–6 | Vectors, dot products, matrices, embeddings | How words become numbers and how numbers get transformed |
| III. Probability | 7–9 | Probability, softmax, entropy, loss | How a model expresses uncertainty and measures its mistakes |
| IV. Calculus and learning | 10–13 | Derivatives, gradients, gradient descent, neural networks | How a model improves itself from data |
| V. The transformer | 14–18 | Attention, transformer blocks, training, text generation | How all the pieces fit into a real LLM |

### How to read the formulas

Formulas are written in a compact language, and like any language it gets easier with practice. When you meet a new symbol, the text around it always says what it means in plain words first. The appendix at the end has a one-page cheat sheet of every symbol used in the book. Keep it open the first time you read.

A good rule: **never skip a formula, but never stare at one either.** Read the sentence that explains it, look at the formula, then look at the worked example. The example is where understanding happens.

### A promise

By the last page you will be able to take a sentence like "The cat sat on the", follow it through every step of a language model, and explain with real math why the model is likely to say "mat" next. Let's begin.
