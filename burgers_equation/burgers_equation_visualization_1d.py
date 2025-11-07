import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# def velocity_calculation(x, t, v):
#     N = 50
    
#     u = torch.zeros_like(x)
    
#     for n in range(1, N + 1):
#         decay = torch.exp(-v * (n * torch.pi) ** 2 * t)
        
#         if n == 1:
#             u += -decay * torch.sin(n * torch.pi * x)
#         else:
#             u += 0 * decay * torch.sin(n * torch.pi * x)
    
#     return u

def velocity_calculation(x, t, v, N_terms=50):
    # x, t, v should be tensors of shape (num_points, 1)
    # Assumes uniform grid for trapz; works for your linspace(100)
    pi = torch.pi
    cos_px = torch.cos(pi * x)
    psi0 = torch.exp((1 - cos_px) / (2 * v * pi))
    
    # Numerical integrals for Fourier coeffs (using trapz on flattened uniform grid)
    x_flat = torch.flatten(x)
    dx = x_flat[1] - x_flat[0]
    a0 = torch.trapz(torch.flatten(psi0), dx=dx)
    
    a = torch.zeros(N_terms + 1, device=x.device, dtype=x.dtype)
    a[0] = a0
    for n in range(1, N_terms + 1):
        cos_nx = torch.cos(n * pi * x)
        integrand = psi0 * cos_nx
        a[n] = 2 * torch.trapz(torch.flatten(integrand), dx=dx)
    
    # Reconstruct psi(x, t)
    psi = a[0].unsqueeze(-1) * torch.ones_like(x)
    for n in range(1, N_terms + 1):
        decay = torch.exp(-v * (n * pi)**2 * t)
        cos_nx = torch.cos(n * pi * x)
        psi += a[n].unsqueeze(-1) * decay * cos_nx
    
    # psi_x
    psi_x = torch.zeros_like(x)
    for n in range(1, N_terms + 1):
        decay = torch.exp(-v * (n * pi)**2 * t)
        sin_nx = torch.sin(n * pi * x)
        psi_x -= a[n].unsqueeze(-1) * decay * (n * pi) * sin_nx
    
    # u = -2 v * psi_x / psi
    u = -2 * v * (psi_x / psi)
    return u


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
    model = BurgersEquation1D().to("mps")
    model.load_state_dict(torch.load("burgers_equation_1d.pt", map_location="mps"), strict=False)
    
    x = torch.linspace(0, 1, 100).unsqueeze(-1)
    v = torch.tensor([0.1]).unsqueeze(-1).repeat(100, 1)
    
    # Setup the figure and axis
    fig, ax = plt.subplots(figsize=(10, 5))
    line_analytical, = ax.plot([], [], label="Analytical Solution", color="blue", linewidth=2)
    line_pred, = ax.plot([], [], label="Neural Network Prediction", linestyle="--", color="red", linewidth=2)
    
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.5, 1.5)  # Adjust based on expected velocity range
    ax.set_xlabel("x")
    ax.set_ylabel("u(x, t)")
    ax.set_title("Burgers' Equation: Analytical vs Neural Network Prediction")
    ax.legend()
    ax.grid()
    
    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, fontsize=12,
                        verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Animation parameters
    num_frames = 100
    duration = 10  # seconds
    
    def init():
        line_analytical.set_data([], [])
        line_pred.set_data([], [])
        time_text.set_text('')
        return line_analytical, line_pred, time_text
    
    def update(frame):
        # Calculate current time (from 0 to 1)
        t_val = frame / (num_frames - 1)
        t = torch.tensor([t_val]).unsqueeze(-1).repeat(100, 1)
        
        # Calculate analytical solution
        u = velocity_calculation(x, t, v)
        
        # Calculate neural network prediction
        with torch.no_grad():
            u_pred = model(x.to("mps"), t.to("mps"), v.to("mps")).cpu()
        
        # Update plot data
        line_analytical.set_data(x.numpy(), u.numpy())
        line_pred.set_data(x.numpy(), u_pred.numpy())
        time_text.set_text(f't = {t_val:.3f}')
        
        # Print error for current frame
        if frame % 10 == 0:
            error = torch.mean((torch.abs(u - u_pred)/torch.abs(u))[1:]).item() * 100
            print(f"Frame {frame}: t={t_val:.3f}, Percentage Error: {error:.2f}%")
        
        return line_analytical, line_pred, time_text
    
    # Create animation
    interval = (duration * 1000) / num_frames  # milliseconds per frame
    ani = FuncAnimation(fig, update, frames=num_frames, init_func=init,
                       blit=True, interval=interval, repeat=True)
    
    plt.show()