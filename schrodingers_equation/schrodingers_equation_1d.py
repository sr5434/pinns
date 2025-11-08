import torch
from torch import nn
import torch.nn.functional as F
import time
from softadapt import SoftAdapt

def schrodinger_calculation(x, t, L, n, m):
    x = x.reshape((-1, 1))
    t = t.reshape((-1, 1))
    h_reduced = 1.0#1.054e-34
    E_n = (h_reduced**2*torch.pi**2*n**2)/(2*m*L**2)
    phi_n = torch.sin((n*torch.pi*x)/L)*(2/L)**0.5
    omega = (E_n*t)/h_reduced
    real = phi_n*torch.cos(omega)
    imag = -phi_n*torch.sin(omega)
    return torch.cat([real, imag], dim=-1)

def calculate_residual_loss(x, t, m, u_v):
    # Make sure that the solution is valid
    u = u_v[:, 0]
    v = u_v[:, 1]

    h_reduced = 1.0#1.054e-34

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]

    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    r_1 = h_reduced*v_t - u_xx*h_reduced**2/(2*m)
    r_2 = h_reduced*u_t + v_xx*h_reduced**2/(2*m)
    # r_1 = h_reduced*u_t - v_xx*h_reduced**2/(2*m)
    # r_2 = h_reduced*v_t + u_xx*h_reduced**2/(2*m)
    return torch.mean(r_1**2)+torch.mean(r_2**2)

def calculate_initial_loss(x, L, n, m):
    # Make sure that the initial states are correct
    pred = model(x, torch.zeros_like(x), n)
    targets = schrodinger_calculation(x, torch.zeros_like(x), L, n, m)
    return F.mse_loss(pred, targets)

def calculate_magnitude_loss(u, L):
    # Make sure the probabilities of a particle being in a specific location sum to 1
    magnitudes = torch.sum(u**2, dim=-1)
    probability_integral = L*torch.mean(magnitudes, dim=0)
    return F.mse_loss(probability_integral, torch.ones_like(probability_integral, device=probability_integral.device))

def magnitude_loss_grouped(psi, L, groups):
    # psi: (N,2); N = groups * points_per_group
    N = psi.shape[0]
    P = N // groups
    psi2 = psi.pow(2).sum(dim=-1).view(groups, P)   # (G, P)
    probs = L * psi2.mean(dim=1)                    # (G,)
    target = torch.ones_like(probs)
    return F.mse_loss(probs, target)

def calculate_boundary_loss(t, L, n):
    # Make sure the states at the edges of the rods are correct
    u_pred_0 = model(torch.zeros_like(t), t, n)
    u_0 = torch.zeros_like(u_pred_0)

    u_pred_1 = model(torch.ones_like(t)*L, t, n)
    u_1 = torch.zeros_like(u_pred_1)

    loss_bc = F.mse_loss(u_pred_0, u_0) + F.mse_loss(u_pred_1, u_1)
    return loss_bc

class SchrodingerEquation1D(nn.Module):
    def __init__(self, L):
        super(SchrodingerEquation1D, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(5, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 2)
        )
        self.L = L

    def forward(self, x, t, n):
        omega = (torch.pi**2 * n**2) / (2.0 * self.L**2)
        t_encoded = torch.cat([t, torch.sin(omega * t), torch.cos(omega * t)], dim=-1)
        inputs = torch.cat([x, t_encoded, (n-1)/2.0], dim=-1)
        raw_output = self.net(inputs)
        psi0_real = torch.sin(n * torch.pi * x / self.L) * (2.0 / self.L)**0.5
        psi0_imag = torch.zeros_like(psi0_real)
        psi0 = torch.cat([psi0_real, psi0_imag], dim=-1)
        # Enforce boundary conditions
        bc = torch.sin(torch.pi * x / self.L)
        return psi0 + torch.tanh(3.0 * t)*bc*raw_output

if __name__ == "__main__":
    device = "mps"
    # Define the model
    examples = 300000000
    steps = 40000# Steps per epoch
    length = 1.0# Length of the 1d box
    mass = 1.0
    batch_size = examples//steps
    epochs = 1
    lr = 1e-3
    decay_steps = 15000
    # softAdapt = SoftAdapt(beta=0.1)
    betas = (0.9, 0.999)
    model = SchrodingerEquation1D(length).to(device)
    # residual_loss_history = []
    # boundary_loss_history = []
    # initial_loss_history = []
    # magnitude_loss_history = []
    # update_loss_weights_every = 5
    optim = torch.optim.Adam(model.parameters(), lr, betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, eta_min=1e-4, T_0 = 5000)
    weights = torch.tensor([1.0, 1.0])
    for i in range(steps):
        start = time.time()
        # Extract batch and ensure gradients are tracked
        # Build a batch: G groups, P x-samples each
        G = batch_size//32
        P = batch_size//G
        max_energy = min(int(i/5000 + 1), 3)
        t_time = torch.rand(G, 1, device=device, requires_grad=True).repeat_interleave(P, dim=0)
        energy_levels = torch.randint(1, max_energy + 1, (G, 1), device=device).repeat_interleave(P, dim=0)
        x_space = torch.rand(G * P, 1, device=device, requires_grad=True)
        # x_space = torch.rand(batch_size, 1, device=device, requires_grad=True)
        # t_time = torch.rand(batch_size, 1, device=device, requires_grad=True)
        # energy_levels = torch.randint(1, 8, (batch_size, 1), device=device)
        output = model(x_space, t_time, energy_levels)

        residual_loss = calculate_residual_loss(x_space, t_time, mass, output)
        # boundary_loss = calculate_boundary_loss(t_time, length, energy_levels)
        # initial_loss = calculate_initial_loss(x_space, length, energy_levels, mass)
        magnitude_loss = magnitude_loss_grouped(output, length, G)#calculate_magnitude_loss(output, length)
        # residual_loss_history.append(residual_loss.item())
        # boundary_loss_history.append(boundary_loss.item())
        # initial_loss_history.append(initial_loss.item())
        # magnitude_loss_history.append(magnitude_loss.item())
        # if (i+1) % update_loss_weights_every == 0:
        #     weights = softAdapt.get_component_weights(torch.tensor(residual_loss_history), torch.tensor(initial_loss_history), torch.tensor(magnitude_loss_history))
        #     residual_loss_history = []
        #     boundary_loss_history = []
        #     initial_loss_history = []
        #     magnitude_loss_history = []
        # weights = softAdapt.get_component_weights(torch.tensor(residual_loss_history), torch.tensor(boundary_loss_history), torch.tensor(initial_loss_history), torch.tensor(magnitude_loss_history))
        loss = weights[0].to(torch.float32).to(device)*residual_loss + weights[1].to(torch.float32).to(device)*magnitude_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()
        optim.zero_grad()
        if i < decay_steps:
            scheduler.step()
        end = time.time()

        print(f"Step: {i+1} | Max Energy: {max_energy} | LR: {scheduler.get_last_lr()[0]} | Total Loss: {loss.item()} | Time: {1000*(end-start)}ms | Residual Loss: {residual_loss.item()} | Magnitude Loss: {magnitude_loss.item()}")
    
    # print("Adam Training complete. Beginning LBFGS training...")
    # build a fixed batch (G,P grid as above)
    # x_fix = torch.linspace(0, length, 256, device=device).repeat(16,1).view(-1,1).requires_grad_(True)
    # t_fix = torch.rand(16,1,device=device).repeat_interleave(256,0).requires_grad_(True)
    # n_fix = torch.randint(1,8,(16,1),device=device).float().repeat_interleave(256,0)

    # opt = torch.optim.LBFGS(model.parameters(), lr=1.0, max_iter=1000, history_size=50, line_search_fn='strong_wolfe')

    # def closure():
    #     opt.zero_grad()
    #     out = model(x_fix, t_fix, n_fix)
    #     res = calculate_residual_loss(x_fix, t_fix, mass, out)
    #     init = calculate_initial_loss(x_fix, length, n_fix, mass)
    #     mag = magnitude_loss_grouped(out, length, 16)
    #     loss = 0.5*res + 0.3*init + 0.2*mag          # static weights for LBFGS
    #     loss.backward()
    #     print(f"LBFGS step loss: {loss.item()}")
    #     return loss

    # for i in range(3000):
    #     opt.step(closure)
    
    example_x = 0.5
    example_t = 0.1
    example_n = 1
    model_input = torch.tensor([[example_x, example_t, example_n]], dtype=torch.float32).to(device)
    model_output = model(model_input[:, 0:1], model_input[:, 1:2], model_input[:, 2:3])[0]
    ground_truth = schrodinger_calculation(model_input[:,0:1], model_input[:,1:2], length, model_input[:,2:3], mass)[0]
    pct_error_real = 100*abs(model_output[0].item() - ground_truth[0].item())/abs(ground_truth[0].item())
    pct_error_imag = 100*abs(model_output[1].item() - ground_truth[1].item())/abs(ground_truth[1].item())
    pct_error = (pct_error_real + pct_error_imag)/2
    print(f"Wave function at x={example_x}, t={example_t}, n={example_n}: {model_output[0]}+{model_output[1]}i, actual: {ground_truth[0]}+{ground_truth[1]}i, percent error: {pct_error}%")

    PATH = "schrodingers_equation_1d.pt"
    torch.save(model.state_dict(), PATH)