import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from Network import Layer_Creation, RELU_Activation, SoftMax_Activation, Container, generate_spiral_data, Categorical_Loss

st.title("Interactive Spiral NN Demo")

# Generate data
X, y = generate_spiral_data(100, 3)

# Sidebar controls
epochs = st.slider("Epochs", 100, 1000, 500, step=50)

learning_rate_input = st.text_input("Learning Rate (0.0001 - 0.1)", "0.001")

try:
    learningRate = float(learning_rate_input)
    if learningRate < 0.0001 or learningRate > 0.1:
        st.warning("Learning rate must be between 0.0001 and 0.1. Using default 0.001.")
        learningRate = 0.001
except ValueError:
    st.warning("Invalid learning rate. Using default 0.001.")
    learningRate = 0.001

# Initialize neural network
container = Container()
container.add(Layer_Creation(2, 64))
container.add(RELU_Activation())
container.add(Layer_Creation(64, 64))
container.add(RELU_Activation())
container.add(Layer_Creation(64, 3))
container.add(SoftMax_Activation())
container.add(Categorical_Loss())

# Forward helper
def forward_only(x, container):
    out = x
    for instance in container.instances:
        if isinstance(instance, Categorical_Loss):
            break
        out = instance.forward_pass(out)
    return out

# Training function
def train_nn(epochs, learningRate):
    losses = []
    for i in range(epochs):
        forward_output = container.forward(X, y)
        losses.append(forward_output)
        container.backward(container.ypreds, y, learningRate)
    return losses

# Decision boundary plotting
def plot_decision_boundary(container, X, y):
    h = 0.02
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = forward_only(grid, container)
    Z = np.argmax(preds, axis=1).reshape(xx.shape)

    fig, ax = plt.subplots()
    ax.contourf(xx, yy, Z, cmap="brg", alpha=0.3)
    ax.scatter(X[:,0], X[:,1], c=np.argmax(y, axis=1), cmap="brg")
    ax.set_title("Decision Boundary")
    return fig

# Run training and plot everything
losses = train_nn(epochs, learningRate)

fig_loss, ax_loss = plt.subplots()
ax_loss.plot(losses)
ax_loss.set_title("Loss Curve")
ax_loss.set_xlabel("Epoch")
ax_loss.set_ylabel("Loss")
st.pyplot(fig_loss)

fig_boundary = plot_decision_boundary(container, X, y)
st.pyplot(fig_boundary)
