import torch
from torch import nn
import torch.nn.functional as F
import time


def calculate_residual_loss(x, t, v, u):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]# First derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]# Second derivative
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    residual =  u_t+u*u_x-u_xx*v
    return torch.mean(residual**2)

class BurgersEquation1D(nn.Module):
    def __init__(self):
        super(BurgersEquation1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(3, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 100),
            nn.Tanh(),
            nn.Linear(100, 1),
        )

    def forward(self, x, t, v):
        inputs = torch.cat([x, t, v], dim=-1)
        raw = self.net(inputs)
        phi = x*(1 - x)
        return -torch.sin(torch.pi * x) + t * raw * phi

if __name__ == "__main__":
    device = "mps"
    # Define the model
    model = BurgersEquation1D().to(device)
    examples = 50000000
    steps = 25000# Steps per epoch
    batch_size = examples//steps
    epochs = 1
    lr = 1e-3
    betas = (0.9, 0.999)
    decay_steps = 15000
    
    optim = torch.optim.Adam(model.parameters(), lr, betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, decay_steps, eta_min=1e-5)
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