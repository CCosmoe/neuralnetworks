# NeuralNetworks Project

## What this is
A neural network built from scratch using only NumPy. No TensorFlow, no PyTorch. The purpose was to deeply understand the math behind neural networks — forward propagation, backpropagation, chain rule, gradient descent — all implemented manually.

## Key files
- **Manual.py** — Single-epoch forward/backward pass that prints every intermediate computation. Educational, step-by-step. No AI used.
- **Network.py** — Dynamic multi-epoch version with a Container class, training loop, loss curve, and decision boundary visualization. No AI used.
- **app.py** — Streamlit frontend with sliders for epochs and learning rate. AI was used to build this file.
- **article.md** — LinkedIn article draft explaining the project. Written to sound like the user's natural voice, not polished/AI-sounding.

## Architecture
Input(2) → Hidden(64, ReLU) → Hidden(64, ReLU) → Output(3, Softmax) → Cross-Entropy Loss
Trained on a spiral dataset (3 classes, 300 points).

## Article status
- Draft is in article.md
- Target audience includes recruiters — tone should be grounded and professional without bragging
- User wants it to sound natural and personal, not like a generic LinkedIn post
- User doesn't want to sound cocky or use slang like "under the hood"
- Opening was rewritten to reflect their real experience: used PyTorch before but didn't understand the math, so they built this to stop guessing
- Clarified that forward propagation class structure came from online sources, but backprop/loss/weight updates were coded by the user after learning concepts through GPT
- Did a cocky-tone pass: softened "the code is all mine", renamed "Things I learned that you don't get from tutorials" to "Things I learned by doing it manually", removed dismissive "I know PyTorch exists"
