import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import torch.functional as F
from torch import nn

class HeatEquation3D(nn.Module):
    def __init__(self):
        super(HeatEquation3D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, 1)
        )

    def forward(self, x, y, z, t, alpha):
        inputs = torch.cat([x, y, z, t, alpha], dim=-1)
        raw = self.net(inputs)
        phi = x*(1 - x)*y*(1 - y)*z*(1 - z)
        return raw * phi

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = HeatEquation3D().to(device)
model.load_state_dict(torch.load("heat_equation_3d.pt", map_location=device))
model.eval()

def heat_calculation(x, y, z, t, alpha):
    return torch.sin(torch.pi * x) * torch.sin(torch.pi * y) * torch.sin(torch.pi * z) * torch.exp(-3*alpha*torch.pi*torch.pi*t)
# Visualization
x = torch.linspace(0, 1, 100, device=device)
y = torch.linspace(0, 1, 100, device=device)
X, Y = torch.meshgrid(x, y, indexing='ij')
alpha = torch.tensor(0.05, device=device, dtype=x.dtype)

X_flat = X.reshape(-1, 1)
Y_flat = Y.reshape(-1, 1)
alpha_flat = torch.full((X_flat.shape[0], 1), alpha.item(), device=device, dtype=X.dtype)

frames = 100
duration_seconds = 10
interval_ms = int(duration_seconds * 1000 / frames)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].set_title("Predicted Temperature Distribution")
axes[1].set_title("Exact Temperature Distribution")

extent = (0, 1, 0, 1)
vmin, vmax = -0.1, 1.0

with torch.no_grad():
    initial_t = torch.tensor(0.0, device=device, dtype=X.dtype)
    initial_t_flat = torch.full((X_flat.shape[0], 1), 0.0, device=device, dtype=X.dtype)
    # Take a slice at z=0.5
    initial_pred = model(X_flat, Y_flat, 0.5*torch.ones_like(Y_flat), initial_t_flat, alpha_flat).reshape(100, 100).cpu().numpy()
    initial_exact = heat_calculation(X, Y, 0.5*torch.ones_like(Y), initial_t, alpha).cpu().numpy()

pred_im = axes[0].imshow(initial_pred, extent=extent, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
exact_im = axes[1].imshow(initial_exact, extent=extent, origin='lower', cmap='hot', vmin=vmin, vmax=vmax)
fig.colorbar(pred_im, ax=axes[0])
fig.colorbar(exact_im, ax=axes[1])
time_text = fig.suptitle("t = 0.00")

def update(frame_index):
    t_value = frame_index / (frames - 1)
    t_scalar = torch.tensor(t_value, device=device, dtype=X.dtype)
    t_flat = torch.full((X_flat.shape[0], 1), t_value, device=device, dtype=X.dtype)
    with torch.no_grad():
        U_pred = model(X_flat, Y_flat, 0.5*torch.ones_like(Y_flat), t_flat, alpha_flat).reshape(100, 100).cpu()
        U_exact = heat_calculation(X, Y, 0.5*torch.ones_like(Y), t_scalar, alpha).cpu()
    pred_im.set_data(U_pred.numpy())
    exact_im.set_data(U_exact.numpy())
    time_text.set_text(f"t = {t_value:.2f}")
    # print(U_pred, U_exact)
    
    return pred_im, exact_im, time_text


ani = animation.FuncAnimation(fig, update, frames=frames, interval=interval_ms, blit=False, repeat=False)

# Save the animation as a video file
Writer = animation.writers['ffmpeg']
writer = Writer(fps=frames//duration_seconds, metadata=dict(artist='Me'), bitrate=1800)
ani.save('heat_equation_3d_visualization.mp4', writer=writer)

plt.show()