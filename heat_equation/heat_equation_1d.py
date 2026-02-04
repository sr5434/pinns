import torch
from torch import nn
import torch.nn.functional as F
import time
import argparse

def heat_calculation(x, t, alpha):
    return torch.sin(torch.pi * x) * torch.exp(-alpha*torch.pi*torch.pi*t)

def calculate_residual_loss(x, t, alpha, u):
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]# First derivative
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]# Second derivative
    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    residual =  u_t-alpha*u_xx
    return torch.mean(residual**2)

def calculate_initial_loss(x, alpha):
    # Calculate loss at t=0
    u_pred = model(x, torch.zeros_like(x), alpha)
    u_0 = heat_calculation(x, torch.zeros_like(x), alpha)
    loss_ic = F.mse_loss(u_pred, u_0)
    return loss_ic

def calculate_boundary_loss(t, alpha):
    # Calculate loss at x=0 and x=1 boundaries
    u_pred_0 = model(torch.zeros_like(t), t, alpha)
    u_0 = heat_calculation(torch.zeros_like(t), t, alpha)

    u_pred_1 = model(torch.ones_like(t), t, alpha)
    u_1 = heat_calculation(torch.ones_like(t), t, alpha)
    
    loss_bc = F.mse_loss(u_pred_0, u_0) + F.mse_loss(u_pred_1, u_1)
    return loss_bc

class HeatEquation1D(nn.Module):
    def __init__(self, hidden_dim=50, hidden_layers=2):
        super(HeatEquation1D, self).__init__()
        layers = [nn.Linear(3, hidden_dim), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.Tanh()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x, t, alpha):
        inputs = torch.cat([x, t, alpha], dim=-1)
        raw = self.net(inputs)
        phi = x*(1 - x)
        return raw * phi

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Heat Equation 1D PINN')
    parser.add_argument('--device', type=str, default='mps', help='Device to use for training (default: mps)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 1e-3)')
    parser.add_argument('--steps', type=int, default=10000, help='Number of training steps (default: 10000)')
    parser.add_argument('--examples', type=int, default=20000000, help='Total number of training examples (default: 20000000)')
    parser.add_argument('--beta1', type=float, default=0.9, help='Adam beta1 parameter (default: 0.9)')
    parser.add_argument('--beta2', type=float, default=0.999, help='Adam beta2 parameter (default: 0.999)')
    parser.add_argument('--hidden-dim', type=int, default=50, help='Hidden layer width (default: 50)')
    parser.add_argument('--hidden-layers', type=int, default=2, help='Number of hidden layers (default: 2)')
    parser.add_argument('--residual-weight', type=float, default=1.0, help='Residual loss weight (default: 1.0)')
    parser.add_argument('--boundary-weight', type=float, default=1.0, help='Boundary loss weight (default: 1.0)')
    parser.add_argument('--initial-weight', type=float, default=2.0, help='Initial condition loss weight (default: 2.0)')
    args = parser.parse_args()
    
    device = args.device
    # Define the model
    model = HeatEquation1D(hidden_dim=args.hidden_dim, hidden_layers=args.hidden_layers).to(device)
    examples = args.examples
    steps = args.steps
    batch_size = examples//steps
    epochs = 1
    lr = args.lr
    betas = (args.beta1, args.beta2)
    
    optim = torch.optim.Adam(model.parameters(), lr, betas)

    for i in range(steps):
        start = time.time()
        # Extract batch and ensure gradients are tracked
        alpha = torch.rand(batch_size, 1, device=device).to(device)
        x_space = torch.rand(batch_size, 1, device=device, requires_grad=True).to(device)
        t_time = torch.rand(batch_size, 1, device=device, requires_grad=True).to(device)

        output = model(x_space, t_time, alpha)

        residual_loss = calculate_residual_loss(x_space, t_time, alpha, output)
        boundary_loss = calculate_boundary_loss(t_time, alpha)
        initial_loss = calculate_initial_loss(x_space, alpha)
        loss = (
            args.residual_weight * residual_loss
            + args.boundary_weight * boundary_loss
            + args.initial_weight * initial_loss
        )

        loss.backward()
        optim.step()
        optim.zero_grad()
        end = time.time()

        print(f"Step: {i+1} | Loss: {loss.item()} | Time: {1000*(end-start)}ms | Residual Loss: {residual_loss.item()} | Boundary Loss: {boundary_loss.item()} | Initial Loss: {initial_loss.item()}")
    
    print("Training complete.")
    example_x = 0.5
    example_t = 0
    example_alpha = 0.1
    model_input = torch.tensor([[example_x, example_t, example_alpha]], dtype=torch.float32).to(device)
    model_output = model(model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3])
    print(f"Predicted temperature at x={example_x}, t={example_t}: {model_output.item()}, actual: {heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}, percent error: {100*abs(model_output.item() - heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item())/heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}%")

    example_x = 0.5
    example_t = 0.1
    example_alpha = 0.1
    model_input = torch.tensor([[example_x, example_t, example_alpha]], dtype=torch.float32).to(device)
    model_output = model(model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3])
    print(f"Predicted temperature at x={example_x}, t={example_t}: {model_output.item()}, actual: {heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}, percent error: {100*abs(model_output.item() - heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item())/heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}%")

    example_x = 0.5
    example_t = 1
    example_alpha = 0.1
    model_input = torch.tensor([[example_x, example_t, example_alpha]], dtype=torch.float32).to(device)
    model_output = model(model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3])
    print(f"Predicted temperature at x={example_x}, t={example_t}: {model_output.item()}, actual: {heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}, percent error: {100*abs(model_output.item() - heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item())/heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}%")

    example_x = 0.5
    example_t = 1
    example_alpha = 0.1
    model_input = torch.tensor([[example_x, example_t, example_alpha]], dtype=torch.float32).to(device)
    model_output = model(model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3])
    print(f"Predicted temperature at x={example_x}, t={example_t}: {model_output.item()}, actual: {heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}, percent error: {100*abs(model_output.item() - heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item())/heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}%")


    example_x = 0.9
    example_t = 1
    example_alpha = 0.1
    model_input = torch.tensor([[example_x, example_t, example_alpha]], dtype=torch.float32).to(device)
    model_output = model(model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3])
    print(f"Predicted temperature at x={example_x}, t={example_t}: {model_output.item()}, actual: {heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}, percent error: {100*abs(model_output.item() - heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item())/heat_calculation(model_input[:,0:1], model_input[:,1:2], model_input[:,2:3]).item()}%")

    PATH = "heat_equation_1d.pt"
    torch.save(model.state_dict(), PATH)
