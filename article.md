# I built a neural network from scratch with just NumPy

I've used PyTorch plenty of times. Call model.fit(), call .backward(), tweak some hyperparameters, etc. It works until it doesn't. When my models weren't converging or the loss was doing something weird, I didn't really know what to change or why. I was just guessing — adjusting learning rates, swapping optimizers, hoping something would stick. I realized I didn't actually understand the math behind what these functions were doing. So I decided to build a neural network from scratch. No PyTorch, no TensorFlow, no autograd. Just NumPy and math.

## What I was trying to do

I picked a spiral dataset that has three classes of points that spiral around each other in 2D. It looks simple but a linear model can't separate spirals. You actually need hidden layers and non-linear activations to get it right. That's what made it a good test. If the backpropagation is wrong even slightly, the network just won't learn and there's no helpful error message telling you what went wrong.

## What I actually built

The network has 2 hidden layers with 64 neurons each, ReLU activations, softmax on the output, and cross-entropy loss. The class structure for forward propagation I picked up from online sources. But everything after that — the loss function, the full backpropagation, the weight updates — I learned conceptually through GPT and coded myself. Most resources I found explain backpropagation with diagrams and equations but not how to actually implement it in code, so that part took a lot of trial and error. Everything is matrix operations in NumPy.

I made two versions. The first one (Manual.py) does a single forward and backward pass and prints every single intermediate value — every matrix multiply, every transpose, every gradient. I did this so I could trace the chain rule with real numbers and actually see what was happening at each step. GPT helped me understand the concepts but I wrote the code myself.

Once I figured out the pattern for how gradients flow backward through each layer, I built the second version (Network.py). This one has a Container class that holds all the layers, runs forward and backward passes automatically, and trains over multiple epochs. Basically I generalized the manual process into something reusable.

## Things I learned by doing it manually

**You need to save the old weights before updating them.** When you're doing backpropagation and you update the output layer's weights first, the hidden layer still needs the original output weights for its gradient calculation. I didn't realize this at first and couldn't figure out why my hidden layer gradients were wrong. Frameworks handle this silently — you'd never even think about it if you're just using PyTorch.

**Softmax will give you infinity if you're not careful.** The exponential function blows up fast. You have to subtract the max value from each row before computing exp() or you get inf values and everything breaks. Small detail, huge impact.

**Bias updates are different from weight updates.** The shapes don't match up so you need to sum across the batch dimension. I spent a while debugging this because the error isn't obvious — the network just doesn't learn properly.

**ReLU's backward pass is basically a gate.** If a neuron output was 0 during forward pass, its gradient gets zeroed out during backward pass. So only the neurons that actually fired get their weights updated. I knew this conceptually but seeing it in the actual numbers made it click.

## How it turned out

500 epochs, learning rate of 0.001, and the network separates all three spirals. The loss drops steadily and the decision boundary plot shows clean non-linear separation between the classes.

I also made a Streamlit app where you can change the epochs and learning rate with sliders and see how it affects the loss curve and decision boundary. I used AI for the frontend part since that wasn't the point of the project — the point was understanding the math and the internals.

## Why I did this

This isn't a production tool and it's not meant to replace any framework.

But now when a model isn't converging I'm not just randomly changing hyperparameters — I'm thinking about what the gradients are actually doing. When I read about optimizers like Adam I understand what problem they're solving because I've hit those limitations myself with basic gradient descent.

I just wanted to stop guessing and actually understand the math behind what I was using. That's it.

Full project is on my GitHub if you want to look at the code or try something similar yourself.
