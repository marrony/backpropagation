# Backpropagation - Neural Network Library in C

A collection of educational programs demonstrating neural networks, backpropagation, and deep learning algorithms implemented from scratch in C.

## Project Overview

This project is an educational deep learning library and demonstration suite that implements:
- **Forward and backward propagation** with automatic gradient computation
- **Multiple neural network architectures**: Feedforward, CNN, Node-based graphs
- **Various activation functions**: Sigmoid, ReLU, Softmax, Linear
- **Matrix operations** for efficient tensor computations
- **MNIST digit classification** examples
- **Reinforcement learning** with Snake game AI

## Core Components

### 1. Neural Network Library (nn.h)

The foundational library implementing core deep learning primitives:

**Matrix Operations:**
- Allocation, initialization, and memory management
- Element-wise operations (add, subtract, multiply, divide)
- Matrix multiplication (forward and transpose variants)
- Dot products and reductions
- Transpose and reshaping

**Activation Functions:**
- **Sigmoid**: `1 / (1 + e^(-x))` with derivative `x(1-x)`
- **ReLU**: `max(0, x)` with derivative `1 if x > 0`
- **Softmax**: Normalizes to probability distribution with numerically stable implementation
- **Linear**: Identity function

**Network Architecture:**
- Layer creation with customizable forward/backward functions
- Weight initialization with randomization
- Network cloning for gradient accumulation
- Forward propagation through multiple layers
- Backward propagation with chain rule

### 2. Node-Based Differentiation Engine (node.h / node.c)

A computation graph system for automatic differentiation:

**Features:**
- Variable nodes and constant nodes
- Arithmetic operations: Add, Subtract, Multiply, Divide
- Neural layers: Linear, Sigmoid, Softmax, ReLU
- Convolutional layers (2D convolution)
- Flatten layer for CNN outputs
- Forward and backward pass through graphs

**Automatic Differentiation:**
- Chain rule propagation through computation graphs
- Gradient accumulation for batch training
- In-place gradient computation

### 3. Backpropagation Demo (back-propagation.c)

Interactive visualization of backpropagation on MNIST:

**Architecture:**
```
Input (28x28) → ReLU (20) → ReLU (10) → Softmax (10)
```

**Features:**
- Real-time training visualization with accuracy graphs
- Per-sample backward propagation
- Cost function: Mean Squared Error (MSE)
- Training/test accuracy monitoring
- Interactive image evaluation

### 4. Convolutional Neural Network (cnn.c / cnn-node.c)

CNN implementation with 2D convolutional layers:

**Architecture:**
```
Input (28x28) → Conv (3×3 kernels, 9 filters) → ReLU → Flatten → Dense (10) → Softmax
```

**Convolution Details:**
- 3×3 convolution kernels
- 9 feature maps (3 filters × 3×3)
- Strided convolution with no padding
- Xavier/Glorot initialization for weights

**Visualization:**
- Real-time display of input images
- Feature map activation visualization
- Filter weights as texture patterns
- Cost function and accuracy metrics

### 5. Snake Game AI (game.c)

Reinforcement learning demonstration using the neural network library:

**Learning Algorithm:**
- Q-learning with experience replay
- Epsilon-greedy exploration strategy
- Bellman's equation for Q-value updates: `Q_new = reward + γ × max(Q_next)`
- Dual memory systems: short-term and long-term training

**Neural Network:**
```
Input (11) → Sigmoid (10) → Softmax (3)
```

**State Representation:**
- Danger detection in all 4 directions
- Current snake direction
- Food relative position

**Actions:** Straight, Left, Right

### 6. Proof of Concept (proof.c)

Step-by-step demonstration of backpropagation mathematics:

Shows the exact chain rule calculations for a simple 3-layer network with explicit forward and backward pass verification.

## Usage

### Building

```bash
# Build all executables
make
```

The Makefile compiles all `.c` files in the root directory into the `bin/` folder.

### Running Programs

```bash
# MNIST backpropagation visualization
./bin/back-propagation

# CNN visualization
./bin/cnn

# CNN with node-based graph
./bin/cnn-node

# Snake AI game
./bin/game

# Node-based tests
./bin/node

# Backpropagation proof
./bin/proof
```

## Technical Details

### Learning Rate Initialization

Both CNN and feedforward networks use Xavier/Glorot initialization:
```
std = sqrt(2 / (fan_in + fan_out))
```

### Loss Functions

**MSE Loss (Regression/CNN):**
```
L = (1/N) × Σ(y_pred - y_true)²
```

**Cross-Entropy with Softmax:**
Used in Snake game with Q-learning for action selection

### Backward Pass Components

1. **Output Layer:**
   - Compute error: `dL/dh = 2(y_pred - y_true)` for MSE
   - Apply activation derivative

2. **Hidden Layers:**
   - Chain rule: `dL/dz = activation'(h) × dL/dh`
   - Propagate gradients backward

3. **Parameter Updates:**
   ```
   W = W - lr × dL/dW
   b = b - lr × dL/db
   ```

### Data Formats

Uses the MNIST IDX format for images and labels:
- `train-images-idx3-ubyte`: 60,000 training images (28x28 grayscale)
- `train-labels-idx1-ubyte`: Training labels (0-9)
- `t10k-images-idx3-ubyte`: 10,000 test images
- `t10k-labels-idx1-ubyte`: Test labels

## Architecture Comparison

| Component | Type | Activation | Use Case |
|-----------|------|------------|----------|
| `nn.h` | Feedforward | Configurable | General purpose |
| `node.h` | Computation Graph | Multiple | Flexible architectures |
| `back-propagation.c` | Feedforward | ReLU + Softmax | MNIST classification |
| `cnn.c` | CNN | ReLU + Softmax | Image features |
| `game.c` | Feedforward | Sigmoid + Softmax | RL agent |

## Key Algorithms Implemented

1. **Forward Propagation**: `h[i] = activation(h[i-1] × W[i] + b[i])`
2. **Backward Propagation**: Chain rule through computation graph
3. **Xavier Initialization**: Weight scaling for stable gradients
4. **Gradient Accumulation**: Batch gradient computation
5. **Experience Replay**: Sequential memory for RL
6. **Bellman Equation**: Q-value updates in Snake game

## Dependencies

- **Raylib**: Graphics library for visualization
- **GNU Make**: Build system
- **POSIX threads**: For parallel operations (optional)

## Educational Value

This project demonstrates:
- Implementation of neural networks from scratch
- Understanding of backpropagation mathematics
- Automatic differentiation via computation graphs
- Convolution operations in CNNs
- Reinforcement learning concepts
- Interactive visualization of learning processes

## License

MIT License
