# Physics-Informed Neural Networks
This repository contains a collection of physics-informed neural networks (PINNs) that are trained to solve problems in quantum mechanics, thermodynamics, and fluid dynamics by learning directly from physics rather than from labeled data. Most of the models can be trained in a few minutes using an M2 Mac. Pretrained models can be found in [this Huggingface model](https://huggingface.co/sr5434/PINN-Collection).

## Table of Contents
- [Getting Started](#getting-started)
- [What is a PINN?](#what-is-a-pinn)
- [Heat Equation](#heat-equation)
- [Burgers' Equation](#burgers-equation)
- [Schrödinger Equation](#schrödinger-equation)
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
python heat_equation_3D.py

# Generate visualizations
python heat_equation_visualizer_3D.py
```
#### Test Pretrained Model on Heat Equation
```bash
cd heat_equation

# Generate visualizations
curl -L "https://huggingface.co/sr5434/PINN-Collection/resolve/main/heat_equation_3d.pt?download=true" -o heat_equation_3d.pt
python heat_equation_visualizer_3D.py
```
#### Train on Burgers' Equation
```bash
# Train the 1D Burgers' equation model
cd burgers_equation
python burgers_equation_1D.py

# Generate visualizations
python burgers_equation_visualization_1D.py
```
#### Test Pretrained Model on Burgers' Equation
```bash
cd burgers_equation

# Generate visualizations
curl -L "https://huggingface.co/sr5434/PINN-Collection/resolve/main/burgers_equation_1d.pt?download=true" -o burgers_equation_1d.pt
python burgers_equation_visualization_1d.py
```
#### Train on Schrödinger Equation(1D)
```bash
# Train the 1D Schrödinger equation model
cd schrodingers_equation
python schrodingers_equation_1d.py

# Generate visualizations
python schrodingers_visualization_1d.py
```

#### Test Pretrained Model on Schrödinger Equation(1D)
```bash
cd schrodingers_equation

# Generate visualizations
curl -L "https://huggingface.co/sr5434/PINN-Collection/resolve/main/schrodingers_equation_1d.pt?download=true" -o schrodingers_equation_1d.pt
python schrodingers_visualization_1d.py
```
#### Train on Schrödinger Equation(Hydrogen)
```bash
# Train the 1D Schrödinger equation model
cd schrodingers_equation
python schrodingers_equation_hydrogen.py

# Generate visualizations
python schrodingers_visualization_hydrogen.py
```

#### Test Pretrained Model on Schrödinger Equation(Hydrogen)
```bash
cd schrodingers_equation

# Generate visualizations
curl -L "https://huggingface.co/sr5434/PINN-Collection/resolve/main/schrodingers_equation_hydrogen.pt?download=true" -o schrodingers_equation_hydrogen.pt
python schrodingers_visualization_hydrogen.py
```

## What is a PINN?
TL;DR: PINNs are neural networks that learn to solve physics problems by learning from the underlying physical laws, rather than from labeled data.


PINNs are just neural networks that approximate functions described by Partial Differential Equations (PDEs). The main thing that is special about PINNs is not their architecture, but rather how they are trained. PINNs are unique because they are trained to satisfy different conditions, unlike most neural networks, which are trained to mimic labeled examples. These conditions are expressed through loss functions, which are detailed below.

### PDE/Physics Loss
The PDE loss function ensures that the model's solution is valid. Essentially, it plugs the model's solution back into the model and compares the left hand side to the right hand side, similar to how a student in Algebra might check their work after solving for a variable. All PINNs must be trained with the PDE loss.

### Boundary Conditions Loss
This loss function checks that the model satisfies boundary conditions, which dictate model behavior at "boundaries". Examples of boundaries are faces of a cube or ends of a rod. Note that this loss is optional if the boundary is enforced mathematically in the model's code (all training scripts in this repository enforce boundary conditions at the model level). All models use Dirichlet boundary conditions, which mandate that the model's output at the boundary is equal to 0.

### Initial Conditions Loss
The initial conditions loss makes sure that the model's outputs at timestep 0 follow the initial conditions of the problem. The main purpose of this is to ensure the model "starts strong", as poor initial performance will only get worse over time. Like the boundary conditions, this loss is optional when the model is designed to always follow the initial conditions of the problem. Only the script for the 1D heat equation uses an initial conditions loss, with the other 2 scripts enforcing initial conditions at the architectural level.

### Total Loss
The loss can be formalized as a weighted sum of the different losses described above:

$$ℒ(\theta) = w_{PDE} * ℒ_{PDE} + w_{BC} * ℒ_{BC} + w_{IC} * ℒ_{IC}$$

The heat equation model places a weight of 2 on the initial conditions loss, while the other models use a weight of 1 for all losses. The weights can be adjusted to prioritize certain losses over others, depending on the problem at hand.

### Note on Units
This implementation uses dimensionless quantities normalized to [0, 1] for numerical stability and generality. The solutions can be scaled to any physical units by applying appropriate transformations. This is standard practice in computational physics and ensures the neural network trains effectively.

## Heat Equation


https://github.com/user-attachments/assets/3d2c5153-d9c6-426f-a677-92130e047e3a



This repository contains code to train PINNs on the 1D, 2D, and 3D heat equations. It also contains code to generate visualizations from the 2D and 3D models (the 3D visualization is just a slice from the middle of a cube). The trained models predict how heat diffuses through a rod, a tile, and a cube respectively. This is the 3D heat equation (for 2D, remove the second derivative w.r.t. z, and for 1D, also remove the second derivative w.r.t. y):

$$\frac{∂u}{∂t} = \alpha\underbrace{(\frac{∂^2u}{∂x^2} + \frac{∂^2u}{∂y^2} + \frac{∂^2u}{∂z^2})}_{\text{Spatial Diffusion/Laplacian}}$$

Here, $\alpha$ is a constant representing the thermal diffusivity of a material. Our model aims to estimate the value of $u(x, y, z)$. The model is trained to support any value of $\alpha$ between 0 and 1, inclusive of the lower and exclusive of the upper. Our model is trained using the Adam optimizer with a fixed learning rate of $1*10^{-3}$. It uses the tanh activation function because it is smooth and continuous. Also, it has a bounded derivative($\frac{d}{dx}tanh(x) = sech^2(x) \in [0, 1]$), which helps prevent exploding gradients. The architectures of the 1D, 2D, and 3D models are the same, ignoring differences in the number of input/output channels. They all have 1 hidden layer, with a 50 dimension hidden state. The 1D model is trained for 10,000 steps and trained on 20,000,000 unlabeled sample inputs (collocation points, timesteps, and values of $\alpha$), while the 2D and 3D models were both trained for 15,000 steps on 75,000,000 sample inputs. I evaluated my model by comparing its results to the results generated by an analytical solution at several points (the points can be seen inside of the training scripts). The 1D model never had more than 1% error, the 2D model did not have more than 1.5% error, and the 3D model got up to 4% error. This jump in error for the 3D model is expected, as it is a much more complicated problem than the 1D and 2D models.


### Applications
- Electrical Engineering: Modeling heat flow in electronics
- Civil Engineering: Designing cooling systems that maximize energy efficiency in buildings
- Food Sciences: Simulating cooking or baking

## Burgers' Equation
<video src="./plots/burgers_equation_1D_visualization.mp4" width="320" height="240" controls></video>


https://github.com/user-attachments/assets/087da0a3-1385-4069-abf7-c7fc6d569190


The repository also contains a script to train a PINN on the 1-dimensional variant of Burgers' Equation, which predicts the instantaneous velocity of a particle in a fluid. Burgers' Equation is as follows:

$$\frac{∂u}{∂t} + u \cdot \nabla u = \nu\nabla^2u$$

Because we are solving the 1D variant of the equation, the gradient simplifies to a single derivative w.r.t. $x$:

$$\frac{∂u}{∂t} + u\frac{∂u}{∂x} = \nu\frac{∂^2u}{∂x^2}$$


$\nu$ represents the viscosity of the fluid, and has the same range as $\alpha$. The model for Burgers' Equation is a deeper and wider version of the one used for the heat equation (2 hidden layers as opposed to one, and 100 hidden dimensions instead of 50). The model supports any value of $\nu$ in the range \[0, 1\). Instead of using a fixed learning rate, the learning rate was decayed following a cosine schedule from $1 \cdot 10^{-3}$ to $1 \cdot 10^{-5}$ over the first 15,000 steps, and then kept flat at $1*10^{-5}$ for an additional 10,000 steps. The model was trained on a total of 50,000,000 examples using the Adam optimizer and tanh activation function. As mentioned earlier, I enforce initial and boundary conditions at the model level, so I only use the PDE loss. The model was evaluated using the visualization script, and percent error was calculated by the frame. The error goes from about 0.5% in earlier frames to 4.5% in later frames.

### Applications
- Public policy: Modeling traffic flow
- Acoustics: Describing sound waves

## Schrödinger Equation
### Time-Dependent Schrödinger Equation in 1D
<video src="./plots/schrodinger_equation_1D.mp4" width="320" height="240" controls></video>

https://github.com/user-attachments/assets/e0f91fa9-b126-4f74-a397-215d36b291f6




There is a script to train a model to predict the wavefunction of a quantum particle in a 1D infinite square well over time, following the Time-Dependent Schrödinger Equation:

$$i\hbar\frac{∂}{∂t}|\psi⟩ = \hat{H}|\psi⟩$$

Where $\hbar$ is a constant representing the reduced Planck constant, defined below:

$$\hbar = \frac{h}{2\pi}$$

$$\hbar \approx 1.0545718*10^{-34}$$

Due to the fact that this is such a small number, a value of 1 is used as a simplification to avoid an underflow. $|\psi⟩$ represents the state vector of the quantum system in bra-ket notation, and is expressed mathematically as a complex wavefunction of space and time. $\hat{H}$ is the Hamiltonian operator, defined for this quantum system as:

$$\hat{H} = \underbrace{-\frac{\hbar}{2m}\frac{∂^2}{∂x^2}}_{\text{Kinetic}} + \underbrace{V(x)}_{\text{Potential}}$$

The Hamiltonian operator returns the total energy of the quantum system(the sum of potential and kinetic energy) given the wavefunction $\psi(x, t)$. In this system, the potential energy, represented by $V(x)$, is 0 inside of the infinite square well because the function models a free particle, and infinite outside of the infinite square well. This infinite potential energy confines the particle to the bounds of the well. The mass of the particle is represented by $m$, and is also set to 1 for simplicity.

The squared magnitude of the quantum wavefunction can be used to estimate the probability density that when observed at a given time, a particle in quantum superposition with a certain energy level will collapse to a given location. Probability density is similar to probability, and the probability that a particle will be observed in the range $[a, b]$ at time $t$ can be calculated by integrating the probability density over that range:

$$P(a \leq x \leq b, t) = \displaystyle\int_{a}^{b} |\psi(x, t)|^2 \,dx$$


The model for Schrödinger Equation is our largest by far, with 4 hidden layers and a hidden size of 256(except for our last hidden layer, which returns a tensor with 128 channels). Also, the model takes in sinusoidal features generated based on the energy level of the particle, as this helps the model adjust to differences in oscillations between lower and higher levels. Due to the oscillatory nature of higher energy levels, the highest level our model supports is 3. The model was trained with a Cosine decay learning rate schedule that had a warm restart whenever the maximum energy level present in the data was increased(the maximum energy level was increased every 15,000 steps, and the warmup occurred slightly before this). The learning rate started at 0.001 and decayed to 0.0001. After 45,000 steps, the learning rate plateaued at the minimum. Like the other 2 models, the tanh activation function and Adam optimizer are used here. When enforcing the initial conditions, the model's raw output is scaled by $tanh(3t)$ instead of $t$. During training, the model had an extra loss, called the magnitude loss. The magnitude loss measures the model's adherence to the normalization condition. It is calculated by taking the mean squared error between the result of the following integral and 1:

$$\displaystyle\int_{0}^{L} |\psi(x, t)|^2 \,dx$$

Here, L is the length of the infinite square well (which is 1 in our dimensionless system). Intuitively, this can be thought of as a metric to verify that the probabilities in the distribution generated by taking the squared magnitude of the wavefunction sum to 100%. This loss is implemented by performing Monte Carlo integration on the squared magnitude of the wavefunction at several points in space, multiplying by the width between points, and comparing that to 1. 

It should also be noted that unlike the other 2 models, this model outputs 2 channels, representing the real and imaginary components of the complex wavefunction.

### Time-Independent Schrödinger Equation for Hydrogen Atom
<video src="./plots/schrodinger_equation_hydrogen.mp4" width="320" height="240" controls></video>

The repository also contains code to train and visualize a PINN that solves the Time-Independent Schrödinger Equation for the radial portion (more on this below) of Hydrogen's wavefunction. The model supports the 1s, 2s, and 2p orbitals. The Time-Independent equation was used because the probability density of hydrogen orbitals does not change over time. The Time-Independent Schrödinger Equation is as follows:

$\hat{H}|\psi⟩ = E|\psi⟩$

Where $E$ represents the energy of the particle, which is constant over time in this scenario. The Hamiltonian operator for the hydrogen atom is defined as:

$$\hat{H} = \underbrace{-\frac{\hbar^2}{2m}\nabla^2}_{\text{Kinetic}} - \underbrace{\frac{e^2}{4\pi\epsilon_0r}}_{\text{Potential}}$$

Where the Laplacian operator in spherical coordinates is defined as:

$$\nabla^2 = \underbrace{\frac{1}{r^2}\frac{∂}{∂r}(r^2\frac{∂}{∂r})}_{\text{Radial}} + \underbrace{\frac{1}{r^2sin\theta}\frac{∂}{∂\theta}(sin\theta\frac{∂}{∂\theta}) + \frac{1}{r^2sin^2\theta}\frac{∂^2}{∂\phi^2}}_{\text{Angular}}$$

And the Coulomb potential term, $-\frac{e^2}{4\pi\epsilon_0r}$, represents the electrostatic attraction between the positively charged nucleus and the negatively charged electron. Here, $e$ is the elementary charge, and $\epsilon_0$ is the permittivity of free space. Similar to the Time-Dependent model, both $\hbar$ and $m$ are set to 1 for simplicity, and we simplify the true Coulomb term to $-\frac{1}{r}$. The Coulomb potential term becomes less negative further from the nucleus, reflecting Coulomb's law that force of attraction becomes stronger as you approach the nucleus. This increased negativity near the nucleus allows wavefunctions concentrated near the center to have lower energy. $\psi(r, \theta, \phi)$ separates into a radial and angular portion as follows:
$$\psi(r, \theta, \phi) = R(r)Y_{lm}(\theta, \phi)$$

$Y_{lm}(\theta, \phi)$ represents the spherical harmonics, which are encoded analytically in our model because they are well-known functions. L and m are quantum numbers. Because our model needs to learn to solve the radial portion, our problem simplifies to an ODE in terms of $u(r) = rR(r)$:

$$-\frac{\hbar}{2}\frac{d^2u}{dr^2} + \frac{l(l+1)\hbar^2}{2r^2}u - \frac{1}{r}u = Eu$$

This system solves the quantum eigenvalue problem for the hydrogen atom, meaning that the analytical values of E are not used in training. The exact values are present in the codebase for evaluation purposes, however. The Rayleigh quotient is used to estimate the energy level of the particle based on the model's current prediction for the wavefunction:

$$E_n \approx \frac{\displaystyle\int_{0}^{\infty} \left(\frac{1}{2}\left(\frac{∂u}{∂r}\right)^2 + \frac{l(l+1)}{2r^2}u^2 - \frac{1}{r}u^2\right) dr}{\displaystyle\int_{0}^{\infty} u^2 dr}$$

To improve the accuracy of the Rayleigh quotient, a loss based on the Virial Theorem is also used during training:

$$(2⟨T⟩ + ⟨V⟩)^2 = 0$$

Where $⟨T⟩$ is the expected kinetic energy, and $⟨V⟩$ is the expected potential energy. The expectation of an operator is calculated like this:

$$⟨A⟩ = \frac{\displaystyle\int_{0}^{\infty} u^*(r)\hat{A}u(r) dr}{\displaystyle\int_{0}^{\infty} u^*(r)u(r) dr}$$

This can be thought of as a continuous average of the operator over all space, weighted by the probability density of the particle being at each location. 

Similar to the model for the TDSE, the magnitude loss is defined as follows:

$$\displaystyle\int_{0}^{\infty} |u(r)|^2  dr$$

To prevent the 2s and 1s orbitals from collapsing to the same state, an orthogonality loss is defined as follows:

$$(\displaystyle\int_{0}^{\infty} u^*_n(r)u_m(r) dr)^2$$

This loss is only enforced between 1s and 2s, and is not enforced for any other orbital combinations.

The model architecture is the same as the one used for the Time-Dependent Schrödinger Equation, except that the input layer has been modified to accept 3 channels instead of 5. The model was trained on 600,000,000 samples using the Adam optimizer with the same cosine schedule as the Time-Dependent model, but without warm restarts. In each step, there were 3 sets of collocation points: general points, central points, and deterministic points. General points were sampled from a Gamma distribution and had a maximum radius of 30. Central points were uniformily sampled within a 3 unit radius to prioritize coverage where most of the action was happening. Deterministic points were sampled from a fixed grid. General and central points were only used for calculating the residual loss, and the deterministic points were used for all other losses. Trapezoidal integration was selected for all integrals because it is more accurate than Monte Carlo integration. Unlike the other models, this model uses spherical coordinates as inputs. The model was evaluated by comparing its results to the analytical solutions for the 1s, 2s, and 2p orbitals. Error is measured with mean absolute error between the predicted and analytical radial wavefunctions.

To use the model to estimate the wavefunction of a dihydrogen cation($H^+_2$), the Linear Combination of Atomic Orbitals (LCAO) method is used. The wavefunction of $H^+_2$ can be approximated as a linear combination of two hydrogen 1s orbitals centered around each nucleus. In nature, the dihydrogen cation is rare, only occuring when a cosmic ray ionizes a hydrogen molecule. However, the fact that it only has one electron makes it a great test case for quantum chemistry simulations, as there is no electron-electron repulsion to account for. 

By default, the visualization script renders all of the Hydrogen orbitals supported by the model and the bonding and antibonding molecular orbitals of $H^+_2$.

### Applications
- Physics Research: Modeling cold atom traps
- Chemistry: Understanding atomic structures and reactions

## Model Performance Summary
| Model | Architecture | Training Samples | Max Error |
|-------|-------------|------------------|-----------|
| Heat 1D | 1 layer, 50 hidden | 20M | <1% |
| Heat 2D | 1 layer, 50 hidden | 75M | <1.5% |
| Heat 3D | 1 layer, 50 hidden | 170M | <4% |
| Burgers' 1D | 2 layers, 100 hidden | 50M | <4.5% |
| Schrödinger 1D | 4 layers, 256/256/128 hidden | 400M | 1% to 6.5% depending on energy level[^1] |
| Schrödinger Hydrogen 1s | 4 layers, 256/256/128 hidden | 600M | <$10^{-4}$ MAE for all cases |

[^1]: The error for the Schrödinger Equation model varies based on the energy level of the particle. Lower energy levels tend to have lower error, while higher energy levels exhibit higher error due to their increased oscillatory behavior.

## Why do you need AI?
It is true that analytical solutions to the heat equation, Burgers' Equation, and Schrödinger Equation are far more efficient than using a PINN. However, there are many unique attributes that make PINNs useful. For example, given the outputs of the model and all spatial/temporal inputs, it is possible to solve for the thermal diffusivity of an object, the viscosity of a fluid, or the energy level of a particle. Also, in more complex scenarios, analytical solutions may not exist, meaning PINNs are the only way to approximate the solution to a PDE. This is especially true in quantum mechanics, where even simple systems like a Helium atom are difficult to solve numerically or analytically. PINNs also have the advantage of being mesh-free, meaning they can make predictions at any point in space and time without needing to be retrained or interpolated.

## What I Learned
I learned a lot about physics and multivariate calculus from doing this project. This project also helped me realize how simple natural concepts like heat diffusion (which require a couple thousand parameters to model) are compared to man-made constructs like language (which require billions or trillions of parameters to model effectively). 

Working on this project brought back some nostalgia for a time when I was very passionate about physics, and it made me feel as if I was reconnecting with my past self.

## What's Next?
 - Scale up Burgers' Equation to 2D and 3D
 - Implement more complex PDEs, such as the Navier-Stokes Equations
 - Experiment with newer optimizers like [Muon](https://kellerjordan.github.io/posts/muon/)
 - Enable inverse problems, where the model solves for physical constants given observations of a system

## Acknowledgements
I want to thank [Krivan Semlani](https://www.linkedin.com/in/krivansemlani/) for inspiring me to work on PINNs and encouraging me to keep up the work.
