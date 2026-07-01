# Building a Neural Network from Scratch in NumPy

I've used PyTorch and TensorFlow before, but the whole thing was a black box to me.

I'd call .forward() and .backward() and the model would (sometimes) train, but I couldn't tell you how the data was actually flowing through the network, how the weights and biases were being updated, or what backpropagation was really doing to any of it.

When something broke I had nothing to debug with, I'd just start guessing at the learning rate or swap optimizer and hope.

So I built one from scratch in NumPy to actually see it.

## What I was trying to do

I picked a spiral dataset that has three classes of points that spiral around each other in 2D. It looks simple but a linear model can't separate spirals.

You actually need hidden layers and non-linear activations to get it right. That's what made it a good test.

If the backpropagation is wrong even slightly, the network just won't learn and there's no helpful error message telling you what went wrong.

## What I actually built

The network has 2 hidden layers with 64 neurons each, ReLU activations, softmax on the output, and cross-entropy loss. Everything is matrix operations in NumPy.

The class structure for forward propagation I picked up from online sources. But everything after that, the loss function, the full backpropagation, the weight updates, I learned conceptually through GPT and coded myself.

Most resources I found explain backpropagation with diagrams and equations that aren't explained well, so that part took a lot of trial and error.

I made two versions. The first one (Manual.py) does a single forward and backward pass and prints every single intermediate value, every matrix multiply, every transpose, every gradient. I did this so I could trace the chain rule with real numbers and actually see what was happening at each step. 

Once I figured out the pattern for how gradients flow backward through each layer, I built the second version (Network.py). This one has a Container class that holds all the layers, runs forward and backward passes automatically, and trains over multiple epochs. Basically I generalized the manual process into something reusable.

Here's the backward pass loop, walking through the network in reverse and updating each layer:

```python

def backward(self, y_pred, y_true, lr):
    gradient = None
    delta = None              # carried gradient before the matmul into weights
    prev_weights = None       # original weights from the layer ahead (needed below)
    output_layer = True
    relu = None

    for instance in reversed(self.instances):
        if isinstance(instance, Categorical_Loss):
            # dL/dZ for softmax + cross-entropy reduces to (y_pred - y_true)
            gradient = instance.backward(y_pred, y_true)
            delta = gradient

        elif isinstance(instance, RELU_Activation):
            # propagate gradient through the next layer's ORIGINAL weights
            gradient = delta @ prev_weights.T
            relu = instance

        elif isinstance(instance, Layer_Creation):
            if not output_layer:
                # ReLU gate: zero gradient where the forward output was 0
                gradient = gradient * relu.backward(instance.z)
                delta = gradient

            # weight gradient: multiply this layer's transposed inputs by its gradient
            dW = np.dot(instance.inputs.T, gradient)
            new_w = instance.weights - lr * dW
            # bias gradient: sum the gradient down the sample axis
            new_b = instance.biases  - lr * np.sum(delta, axis=0, keepdims=True)

            # save OLD weights before overwriting, the next layer back needs them
            _, _, prev_weights, _ = instance.updating_weights_biases(new_w, new_b)
            output_layer = False
            
```

Iterating in reverse meant each layer's update depended on the *original* weights of the layer ahead of it, which is why every `Layer_Creation` keeps an `old_weights` reference. That detail is what I called out in the lessons below.

## Things I learned by doing it manually

**You need to save the old weights before updating them.** When you're doing backpropagation and you update the output layer's weights first, the hidden layer still needs the original output weights for its gradient calculation. In the backward pass snippet above, that's the `updating_weights_biases` line at the bottom of the loop. It returns `prev_weights` so the next iteration can use them before they get overwritten.

I didn't realize this at first and couldn't figure out why my hidden layer gradients were wrong. Frameworks handle this silently. You'd never even think about it if you're just using PyTorch.

**Softmax will give you infinity if you're not careful.** The softmax equation is `exp(z) / sum(exp(z))`. The problem is exp() grows extremely fast. Something like exp(300) is already infinity.

So before computing exp(), you subtract the max value from each row: `exp(z - max) / sum(exp(z - max))`. This shifts all the numbers down so the largest one becomes 0, and exp(0) is just 1.

The probabilities still come out the same because you're dividing by the sum. Subtracting the same constant from every value doesn't change the ratios, it just keeps the numbers small enough that exp() doesn't overflow.

**Bias updates are different from weight updates.** During forward propagation, the bias is a single row that gets added to every sample in the batch. So its shape changes: it goes from `(3,)` to being part of a `(samples, 3)` matrix.

When you backpropagate, the gradient comes back in that expanded shape with a row per sample, but the bias is still just `(3,)`. To get the gradient back to the right shape, you sum each column: each column represents a single neuron, and every sample in that column shared the same bias value.

Then you can actually update: `b = b - learning_rate * db`, where `b` is the actual bias and `db` is the summed gradient.

This one took me a while to figure out because if you try to update a `(3,)` bias with a `(samples, 3)` gradient without summing first, NumPy doesn't throw an error, it just broadcasts and gives you the wrong result. The network runs fine, it just quietly doesn't learn.

**ReLU's backward pass is basically a gate.** If a neuron output was 0 during forward pass, its gradient gets zeroed out during backward pass. So only the neurons that actually fired get their weights updated. I knew this conceptually but seeing specific rows and columns that were already 0 go through an equation and stay 0 is what made it click.

**Gradient descent finally made sense when I saw the numbers.** The update equation is `w = w - learning_rate * gradient`. I never really understood why that worked until I watched it happen step by step.

If a gradient is positive, that means increasing that weight would increase the loss. The negative sign in the equation flips it, so the weight moves in the opposite direction, toward lower loss. If the gradient is negative, the weight gets pushed up instead.

That's all gradient descent is doing: using the sign and size of each gradient to figure out which way to move every weight to minimize loss. I only got this by printing the actual numbers and seeing how they changed.

## How it turned out

500 epochs, learning rate of 0.001, and the network separates all three spirals. The loss drops steadily and the decision boundary plot shows clean non-linear separation between the classes.

Getting there took a fair amount of guessing on the learning rate because one learning rate for the whole network only gets you so far. Higher and the loss would blow up. Lower and it barely moved. 0.001 was the one that ended up working, but there was no real reason I picked it other than trying numbers until one did.

I also made a Streamlit app where you can change the epochs and learning rate with sliders and see how it affects the loss curve and decision boundary. I used AI for the frontend part since that wasn't the point of the project, the point was understanding the math and the internals.

Full project is on my GitHub if you want to look at the code. If you've built one from scratch too, what surprised you most when you actually did the math by hand?

#MachineLearning #NeuralNetworks #Python #NumPy #DeepLearning
