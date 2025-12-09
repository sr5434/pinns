# Physics-Informed Neural Networks
This repository contains a collection of physics-informed neural networks (PINNs) that are trained to solve problems in quantum mechanics, thermodynamics, and fluid dynamics by learning directly from physics rather than from labeled data. All of the models can be trained in a few minutes using an M2 Mac. Pretrained models can be found in [this Huggingface model](https://huggingface.co/sr5434/PINN-Collection).

## Table of Contents
- [Getting Started](#getting-started)
- [What is a PINN?](#what-is-a-pinn)
- [Heat Equation](#heat-equation)
- [Burgers' Equation](#burgers-equation)
- [Schrödinger's Equation](#schrödingers-equation)
- [Model Performance Summary](#model-performance-summary)
- [Why do you need AI?](#why-do-you-need-ai)
- [What I Learned](#what-i-learned)
- [What's Next?](#whats-next)
- [Acknowledgements](#acknowledgements)

## Getting Started

### Prerequisites
- Python 3.12+
- PyTorch
- NumPy
- Matplotlib

### Installation
```bash
git clone https://github.com/sr5434/pinns.git
cd pinns
pip install -r requirements.txt
```

### Usage
#### Train on Heat Equation
```bash
# Train the 3D heat equation model
cd heat_equation
python heat_equation_3d.py

# Generate visualizations
python heat_equation_visualizer_3d.py
```
#### Test Pretrained Model on Heat Equation
```bash
cd heat_equation

# Generate visualizations
curl "https://huggingface.co/sr5434/PINN-Collection/resolve/main/heat_equation_3d.pt?download=true" -o heat_equation_3d.pt
python heat_equation_visualizer_3d.py
```
#### Train on Burgers' Equation
```bash
# Train the 1D Burgers' equation model
cd burgers_equation
python burgers_equation_1d.py

# Generate visualizations
python burgers_equation_visualization_1d.py
```
#### Test Pretrained Model on Burgers' Equation
```bash
cd burgers_equation

# Generate visualizations
curl "https://huggingface.co/sr5434/PINN-Collection/resolve/main/burgers_equation_1d.pt?download=true" -o burgers_equation_1d.pt
python burgers_equation_visualization_1d.py
```
#### Train on Schrödinger's Equation
```bash
# Train the 1D Schrödinger's equation model
cd schrodingers_equation
python schrodingers_equation_1d.py

# Generate visualizations
python schrodingers_visualization_1d.py
```

#### Test Pretrained Model on Schrödinger's Equation
```bash
cd schrodingers_equation

# Generate visualizations
curl "https://huggingface.co/sr5434/PINN-Collection/resolve/main/schrodingers_equation_1d.pt?download=true" -o schrodingers_equation_1d.pt
python schrodingers_visualization_1d.py
```

## What is a PINN?
TL;DR: PINNs are neural networks that learn to solve physics problems by learning from the underlying physical laws, rather than from labeled data.


PINNs are just neural networks that approximate functions described by Partial Differential Equations (PDEs). The main thing that is special about PINNs is not their architecture, but rather how they are trained. PINNs are unique because they are trained to satisfy different conditions, unlike most neural networks, which are trained to mimic labeled examples. These conditions are expressed through loss functions, which are detailed below.

### PDE/Physics Loss
The PDE loss function ensures that the model's solution is valid. Essentially, it plugs the model's solution back into the model and compares the left hand side to the right hand side, similar to how a student in Algebra might check their work after solving for a variable. All PINNs must be trained with the PDE loss.

### Boundary Conditions Loss
This loss function checks that the model satisfies boundary conditions, which dictate model behavior at "boundaries". Examples of boundaries are faces of a cube or ends of a rod. Note that this loss is optional if the boundary is enforced mathematically in the model's code (all training scripts in this repository enforce boundary conditions at the model level). 

### Initial Conditions Loss
The initial conditions loss makes sure that the model's outputs at timestep 0 follow the initial conditions of the problem. The main purpose of this is to ensure the model "starts strong", as poor initial performance will only get worse over time. Like the boundary conditions, this loss is optional when the model is designed to always follow the initial conditions of the problem. Only the script for the 1d Burgers' Equation enforces initial conditions at the architectural level, so all other scripts use an initial conditions loss.

### Note on Units
This implementation uses dimensionless quantities normalized to [0, 1] for numerical stability and generality. The solutions can be scaled to any physical units by applying appropriate transformations. This is standard practice in computational physics and ensures the neural network trains effectively.

## Heat Equation
<video src="./plots/heat_equation_3d_visualization.mp4" width="320" height="240" controls></video>


https://github.com/user-attachments/assets/60dc2456-4292-490c-99aa-bce706ce7bb7


This repository contains code to train PINNs on the 1d, 2d, and 3d heat equations. It also contains code to generate visualizations from the 2d and 3d models (the 3d visualization is just a slice from the middle of a cube). The trained models predict how heat diffuses through a rod, a tile, and a cube respectively. This is the 3d heat equation (for 2d, remove the second derivative w.r.t. z, and for 1d, also remove the second derivative w.r.t. y):

$\frac{∂u}{∂t} = \alpha(\frac{∂^2u}{∂x^2} + \frac{∂^2u}{∂y^2} + \frac{∂^2u}{∂z^2})$

Here, $\alpha$ is a constant representing the thermal diffusivity of a material. Our model aims to estimate the value of $u(x, y, z)$. The model is trained to support any value of $\alpha$ between 0 and 1, inclusive of the lower and exclusive of the upper. Our model is trained using the Adam optimizer with a fixed learning rate of $1*10^{-3}$. It uses the tanh activation function. The architectures of the 1d, 2d, and 3d models are the same, ignoring differences in the number of input/output channels. They all have 1 hidden layer, with a 50 dimension hidden state. The 1d model is trained for 10,000 steps and trained on 20,000,000 sample inputs (unlabeled random coordinates, timesteps, and values of $\alpha$), while the 2d and 3d models were both trained for 15,000 steps on 75,000,000 sample inputs. I evaluated my model by comparing its results to the results generated by an analytical solution at several points (the points can be seen inside of the training scripts). The 1d model never had more than 1% error, the 2d model did not have more than 1.5% error, and the 3d model got 10% error in one test case, but got less than 5% for the rest of the test points. This jump in error for the 3d model is expected, as it is a much more complicated problem than the 1d and 2d models. Also, the 10% error occurred at the last timestep, meaning that errors had compounded over previous timesteps.


### Applications
- Electrical Engineering: Modeling heat flow in electronics
- Civil Engineering: Designing cooling systems that maximize energy efficiency in buildings
- Food Sciences: Simulating cooking or baking

## Burgers' Equation
<video src="./plots/burgers_equation_1d_visualization.mp4" width="320" height="240" controls></video>


https://github.com/user-attachments/assets/087da0a3-1385-4069-abf7-c7fc6d569190


The repository also contains a script to train a PINN on the 1-dimensional variant of Burgers' Equation (higher dimensions coming soon!), which predicts the instantaneous velocity of a particle in a fluid. This is the equation for 1d fluids:

$\frac{∂u}{∂t} + u\frac{∂u}{∂x} = \nu\frac{∂^2u}{∂x^2}$


$\nu$ represents the viscosity of the fluid, and has the same range as $\alpha$. The model for Burgers' Equation is a deeper and wider version of the one used for the heat equation (2 hidden layers as opposed to one, and 100 hidden dimensions instead of 50). The model supports any value of $\nu$ in the range \[0, 1\). Instead of using a fixed learning rate, the learning rate was decayed following a cosine schedule from $1\*10^{-3}$ to $1\*10^{-5}$ over the first 15,000 steps, and then kept flat at $1*10^{-5}$ for an additional 10,000 steps. The model was trained on a total of 50,000,000 examples. As mentioned earlier, I enforce initial and boundary conditions at the model level, so I only use the PDE loss. The model was evaluated using the visualization script, and percent error was calculated by the frame. The error goes from about 0.5% in earlier frames to 4.5% in later frames.

### Applications
- Public policy: Modeling traffic flow
- Acoustics: Describing sound waves

## Schrödinger's Equation
<video src="./plots/schrodinger_equation_1d.mp4" width="320" height="240" controls></video>

https://github.com/user-attachments/assets/e0f91fa9-b126-4f74-a397-215d36b291f6




There is a script to train a model to predict the wavefunction of a quantum particle in a 1d box(infinite square well) over time, following the Time-Dependent Schrödinger's Equation:

$i\hbar\frac{∂\psi}{∂t} = \hat{H}\psi(x, t)$

Where $\hbar$ is a constant representing the reduced Planck constant, defined below:

$\hbar = \frac{h}{2\pi}$

$\hbar \approx 1.0545718*10^{-34}$

Due to the fact that this is such a small number, a value of 1 is used as a simplification to avoid an underflow. $\hat{H}$ is the Hamiltonian operator, defined for this quantum system as:

$\hat{H} = -\frac{\hbar}{2m}\frac{∂^2}{∂x^2} + V(x)$

The Hamiltonian operator returns the total energy of the quantum system(the sum of potential and kinetic energy) given the wavefunction $\psi(x, t)$. In this system, the potential energy, represented by $V(x)$, is 0 inside of the box because the particle can move freely, and infinite outside of the box because the particle cannot exist there.

The squared magnitude of the quantum wavefunction can be used to estimate the probability density that when observed at a given time, a particle in quantum superposition with a certain energy level will collapse to a given location. Probability density is similar to probability, and the probability that a particle will be observed in the range $[a, b]$ at time $t$ can be calculated by integrating the probability density over that range:

$P(a \leq x \leq b, t) = \displaystyle\int_{a}^{b} |\psi(x, t)|^2 \,dx$


The model for Schrödinger's Equation is our largest by far, with 4 hidden layers and a hidden size of 256(except for our last hidden layer, which returns a tensor with 128 channels). Also, the model takes in sinusoidal features generated based on the energy level of the particle, as this helps the model adjust to differences in oscillations between lower and higher levels. Due to the oscillatory nature of higher energy levels, the highest level our model supports is 3. The model was trained with a Cosine decay learning rate schedule that had a warm restart whenever the maximum energy level present in the data was increased(the maximum energy level was increased every 15,000 steps, and the warmup occurred slightly before this). The learning rate started at 0.001 and decayed to 0.0001. After 45,000 steps, the learning rate plateaued at the minimum. When enforcing the initial conditions, the model's raw output is scaled by $tanh(3t)$ instead of $t$. During training, the model had an extra loss, called the magnitude loss. The magnitude loss measures the model's adherence to the normalization condition. It is calculated by taking the mean squared error between the result of the following integral and 1:

$\displaystyle\int_{0}^{L} |\psi(x, t)|^2 \,dx$

Here, L is the length of the box (which is 1 in our dimensionless system). Intuitively, this can be thought of as a metric to verify that the probabilities in the distribution generated by taking the squared magnitude of the wavefunction sum to 1. This loss is implemented by performing Monte Carlo integration on the squared magnitude of the wavefunction at several points in space, multiplying by the width between points, and comparing that to 1. 

### Applications
- Modeling strongly cold atom traps
- Nanotechnology: Predicting the activity of strongly confined quantum dots

## Model Performance Summary
| Model | Architecture | Training Samples | Max Error |
|-------|-------------|------------------|-----------|
| Heat 1D | 1 layer, 50 hidden | 20M | <1% |
| Heat 2D | 1 layer, 50 hidden | 75M | <1.5% |
| Heat 3D | 1 layer, 50 hidden | 75M | <10% |
| Burgers' 1D | 2 layers, 100 hidden | 50M | <4.5% |
| Schrödinger's 1D | 4 layers, 256/256/128 hidden | 400M | 1% to 6.5% depending on energy level[^1] |

[^1]: The error for the Schrödinger's Equation model varies based on the energy level of the particle. Lower energy levels tend to have lower error, while higher energy levels exhibit higher error due to their increased oscillatory behavior.

## Why do you need AI?
It is true that analytical solutions to the heat equation, Burgers' Equation, and Schrödinger's Equation are far more efficient than using a PINN. However, there are many unique attributes that make PINNs useful. For example, given the outputs of the model and all spatial/temporal inputs, it is possible to solve for the thermal diffusivity of an object, the viscosity of a fluid, or the energy level of a particle. Also, in more complex scenarios, analytical solutions may not exist, meaning PINNs are the only way to approximate the solution to a PDE. This is especially true in quantum mechanics, where even simple systems like a Helium atom are difficult to solve numerically or analytically. PINNs also have the advantage of being mesh-free, meaning they can make predictions at any point in space and time without needing to be retrained or interpolated.

## What I Learned
I learned a lot about physics and multivariate calculus from doing this project. This project also helped me realize how simple natural concepts like heat diffusion (which require a couple thousand parameters to model) are compared to man-made constructs like language (which require billions or trillions of parameters to model effectively). 

Working on this project brought back some nostalgia for a time when I was very passionate about physics, and it made me feel as if I was reconnecting with my past self.

## What's Next?
 - Scale up Burgers' Equation to 2d and 3d
 - Implement more complex PDEs, such as the Navier-Stokes Equations
 - Experiment with newer optimizers like [Muon](https://kellerjordan.github.io/posts/muon/)
 - Enable inverse problems, where the model solves for physical constants given observations of a system

## Acknowledgements
I want to thank [Krivan Semlani](https://www.linkedin.com/in/krivansemlani/) for inspiring me to work on PINNs and encouraging me to keep up the work.
