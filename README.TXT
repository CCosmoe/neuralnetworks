Neural Network from Scratch (NumPy)

Overview

- This project is a neural network built entirely from scratch using only NumPy. The goal was to deeply understand how neural 
networks operate under the hood by manually implementing forward propagation and backpropagation, without relying on high-level 
libraries such as TensorFlow or PyTorch.

- The project starts with a single forward–backward pass (one epoch. Code can be found in Manual.py), and later evolves into a 
dynamic neural network capable of training over multiple epochs(Code can be found in Network.py).

Features
Forward Propagation
- Implemented weights and biases manually as NumPy arrays.
- Created a custom activation function (ReLU) to introduce non-linearity.
- Created a softmax function for multi-class classification.
- Produced final predictions as probability distributions.


Backpropagation
- Derived gradients manually using the chain rule.
- Calculated error via cross-entropy loss function(manually created).
- Updated weights and biases step by step using gradient descent.
- Ensured the math aligned with theoretical formulas(GPT helped a lot with conceptual understanding) for each derivative.


Training
- Started with a single-epoch implementation, where forward and backward passes were computed once manually.
- Extended into a dynamic neural network:
- Allowed multiple epochs of training.
- Loss decreased over time as weights adjusted.
- Showcased how neural networks learn patterns from data through repeated updates.

Key Concepts Implemented
- Weights & Biases – learned parameters initialized randomly.
- Activation Functions – introduced non-linear decision boundaries.
- Softmax Function – converted logits into probability distributions.
- Cross-Entropy Loss – measured the difference between predicted and true labels.
- Chain Rule – applied to compute gradients in backpropagation.
- Gradient Descent – optimization loop updating weights & biases.
- Epochs – training loop to repeatedly refine the network.


Why This Project?
By building each component from scratch, I gained a strong understanding of:
- How data flows through layers in forward propagation.
- How gradients are computed and applied in backpropagation.
- How repeated epochs lead to learning and loss minimization.
- This foundation makes it easier to understand and debug neural networks in high-level frameworks later.
