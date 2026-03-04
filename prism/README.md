# Reproducing OpenAI's Prism

[Prism](https://openai.com/prism/) is a LaTeX-native workspace that integrates language models within scientific writing. It has features like:

- Autoformalization, using interactive proven theorem (e.g. Coq, Lean) and proofreading.
- Transforming code into pseudocode and vice versa.
- Citation and literature search.

We reproduce Prism-like loops locally simply using automatic TeX compilation triggers by user or agent file saving. This allows real-time interactivity:

```shell
ls *.tex | entr -c latexmk -pdf -interaction=nonstopmode <filename>.tex
```