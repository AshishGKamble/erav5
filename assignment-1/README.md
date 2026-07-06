# Assignment 1 - Four proofs about neural networks

A single-page site that proves four claims about neural networks. Every model is
a tiny neural network written from scratch in plain JavaScript, with no external
libraries. Nothing is precomputed: each section trains live in the browser when
it scrolls into view.

## The four claims

- **S1-1 Activations exist for a reason.** A linear model draws a straight line
  and is stuck near chance on two concentric rings. One ReLU hidden layer wraps
  the ring to ~99%.
- **S1-2 Depth without nonlinearity is a lie.** Five stacked linear layers
  collapse to a single linear map, proven by multiplying the weight matrices.
  ReLU between the same layers breaks the tie.
- **S1-3 Embeddings learn similarity from nothing but next-token.** Trained only
  to predict the next token in a toy grammar, the embedding table clusters
  same-category tokens, though similarity was never supplied.
- **S1-4 Memorization vs generalization.** An over-parameterized net memorizes at
  n=20 (train high, test low) and the gap closes as the dataset grows to n=2000.

## Run locally

Because the code uses ES modules, open it through a local web server rather than
the `file://` protocol:

```
cd assignment-1
python3 -m http.server 8000
# then open http://localhost:8000
```

## Project layout

```
index.html          page shell and copy
styles.css          clean scientific theme
js/nn.js            neural network library (matrices, MLP, Adam, PCA)
js/viz.js           canvas plotting helpers
js/demo-rings.js    S1-1
js/demo-depth.js    S1-2
js/demo-embed.js    S1-3
js/demo-general.js  S1-4
js/app.js           wires demos to the page, lazy-starts on scroll
netlify.toml        static deploy config
```

## Deploy

The site is fully static. Any static host works. For Netlify:

```
netlify deploy --dir=. --prod
```
