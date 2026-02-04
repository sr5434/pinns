import torch
from torch import nn
import torch.nn.functional as F
import time
import argparse


def calculate_residual_loss(x, t, v, u):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]# First derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]# Second derivative
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    residual =  u_t+u*u_x-u_xx*v
    return torch.mean(residual**2)

class BurgersEquation1D(nn.Module):
    def __init__(self, hidden_dim=100, hidden_layers=3):
        super(BurgersEquation1D, self).__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, v):
        inputs = torch.cat([x, t, v], dim=-1)
        raw = self.net(inputs)
        phi = x*(1 - x)
        return -torch.sin(torch.pi * x) + t * raw * phi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Burgers Equation 1D PINN')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training (default: mps)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--steps', type=int, default=25000, help='Number of training steps (default: 25000)')
    parser.add_argument('--examples', type=int, default=50000000, help='Total number of training examples (default: 50000000)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 parameter (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 parameter (default: 0.999)')
    parser.add_argument('--decay-steps', type=int, default=15000, help='Number of steps for learning rate decay (default: 15000)')
    parser.add_argument('--eta-min', type=float, default=1e-5, help='Minimum learning rate for scheduler (default: 1e-5)')
    parser.add_argument('--hidden-dim', type=int, default=100, help='Hidden layer width (default: 100)')
    parser.add_argument('--hidden-layers', type=int, default=3, help='Number of hidden layers (default: 3)')
    args = parser.parse_args()
    
    device = args.device
    # Define the model
    model = BurgersEquation1D(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    examples = args.examples
    steps = args.steps
    batch_size = examples//steps
    epochs = 1
    lr = args.lr
    betas = (args.beta1, args.beta2)
    decay_steps = args.decay_steps
    
    optim = torch.optim.Adam(model.parameters(), lr, betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, decay_steps, eta_min=args.eta_min)
    for i in range(steps):
        start = time.time()
        # Extract batch and ensure gradients are tracked
        viscosity = torch.rand(batch_size, 1, device=device).to(device)
        x_space = torch.rand(batch_size, 1, device=device, requires_grad=True).to(device)
        t_time = torch.rand(batch_size, 1, device=device, requires_grad=True).to(device)

        output = model(x_space, t_time, viscosity)
        residual_loss = calculate_residual_loss(x_space, t_time, viscosity, output)

        residual_loss.backward()
        optim.step()
        optim.zero_grad()
        if i < decay_steps:
            scheduler.step()
        end = time.time()

        print(f"Step: {i+1} | LR: {optim.param_groups[0]['lr']} | Loss: {residual_loss.item()} | Time: {1000*(end-start)}ms")
    
    print("Training complete.")
    PATH = "burgers_equation_1d.pt"
    torch.save(model.state_dict(), PATH)
