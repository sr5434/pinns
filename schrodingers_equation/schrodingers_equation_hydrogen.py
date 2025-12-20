import torch
from torch import nn
import torch.nn.functional as F
import time
# from softadapt import SoftAdapt

def calculate_residual_loss(radial, theta, phi, n, m, u):
    # Make sure that the solution is valid

    h_reduced = 1.0#1.054e-34

    energies = -1/(2*(n**2))

    # Coulomb potential term
    V = -1.0 / radial

    # Laplacian scaling factor
    laplacian_factor = h_reduced**2 / (2 * m)

    # Compute Laplacian
    # Beware: this gets very hairy
    
    # Radial term
    u_r = torch.autograd.grad(u, radial, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, radial, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
    radial_term = u_rr+(2/radial)*u_r

    # Polar term
    u_theta = torch.autograd.grad(u, theta, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_thetatheta = torch.autograd.grad(u_theta, theta, grad_outputs=torch.ones_like(u_theta), create_graph=True)[0]
    polar_term = (1/(radial**2))*(u_thetatheta + u_theta*torch.cos(theta)/torch.sin(theta))

    # Azimuthal term
    u_phi = torch.autograd.grad(u, phi, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_phiphi = torch.autograd.grad(u_phi, phi, grad_outputs=torch.ones_like(u_phi), create_graph=True)[0]
    azimuthal_term = (1/(radial**2 * torch.sin(theta)**2))*u_phiphi

    hamiltonian_u = -laplacian_factor * (radial_term + polar_term + azimuthal_term) + V*u

    r = hamiltonian_u - energies * u
    return torch.mean(r**2)

def magnitude_loss_grouped(psi, radial, theta, groups, r_max):
    # psi: (N,2); N = groups * points_per_group
    N = psi.shape[0]
    P = N // groups
    prob_density = psi.pow(2)
    volume = radial**2 * torch.sin(theta)
    psi2 = (prob_density * volume).view(groups, P)
    sampling_volume = 2.0 * r_max * torch.pi**2
    integral = sampling_volume * psi2.mean(dim=1)
    target = torch.ones_like(integral)
    return F.mse_loss(integral, target)

def orthogonality_loss(psi, radial, theta, n, groups):
    N = psi.shape[0]
    P = N // groups
    psi_reshaped = psi.view(groups, P)
    volume = radial**2 * torch.sin(theta)
    overlaps = []
    for i in range(groups):
        for j in range(i+1, groups):
            # Only compute overlap for different energy levels
            if n[i*P] != n[j*P]:
                prod = psi_reshaped[i] * psi_reshaped[j] * volume
                sampling_volume = 2.0 * radial.max() * torch.pi**2
                integral = sampling_volume * prod.mean()
                overlaps.append(integral)
    if len(overlaps) == 0:
        return torch.tensor(0.0, device=psi.device)
    overlaps_tensor = torch.stack(overlaps)
    # target = torch.zeros_like(overlaps_tensor)
    return torch.mean(overlaps_tensor**2)#F.mse_loss(overlaps_tensor, target)



class SchrodingerEquationHydrogen(nn.Module):
    def __init__(self, L):
        super(SchrodingerEquationHydrogen, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(4, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        self.L = L

    def forward(self, radial, theta, phi, n):
        # omega = (torch.pi**2 * n**2) / (2.0 * self.L**2)
        inputs = torch.cat([radial, theta, phi, (n-1)/2.0], dim=-1)
        raw_output = self.net(inputs)
        # Enforce boundary conditions
        bc = torch.exp(-radial/n)
        return bc*raw_output

if __name__ == "__main__":
    device = "mps"
    # Define the model
    examples = 10000000
    steps = 1000 # Steps per epoch
    length = 1.0 # Length of the 1d box
    mass = 1.0
    batch_size = examples//steps
    max_energy = 1
    r_max = 20.0
    theta_max = torch.pi
    phi_max = 2*torch.pi
    n_groups = 500 # Higher number of groups -> better magnitude loss approximation
    epochs = 1
    lr = 1e-3
    betas = (0.9, 0.999)
    model = SchrodingerEquationHydrogen(length).to(device)
    optim = torch.optim.Adam(model.parameters(), lr, betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, eta_min=1e-5, T_max=steps)#CosineAnnealingWarmRestarts(optim, eta_min=1e-4, T_0 = 15000)
    
    for i in range(steps):
        start = time.time()
        # min(10.0-i/steps, 1.0)
        weights = torch.tensor([1.0, 1.0, 1.0], device=device) # Residual, Magnitude, Orthogonality
        # Extract batch and ensure gradients are tracked
        # Build a batch: G groups, P x-samples each
        G = batch_size//n_groups
        P = batch_size//G
        energy_levels = torch.randint(1, max_energy + 1, (G, 1), device=device).repeat_interleave(P, dim=0)
        epsilon = 1e-6 # to avoid zero radius/theta
        radial = torch.rand(G * P, 1, device=device, requires_grad=True)*(r_max - epsilon) + epsilon
        theta = torch.rand(G * P, 1, device=device, requires_grad=True)*(theta_max - epsilon) + epsilon
        phi = torch.rand(G * P, 1, device=device, requires_grad=True)*phi_max
        # x_space = torch.rand(batch_size, 1, device=device, requires_grad=True)
        # t_time = torch.rand(batch_size, 1, device=device, requires_grad=True)
        # energy_levels = torch.randint(1, 8, (batch_size, 1), device=device)
        output = model(radial, theta, phi, energy_levels)
        residual_loss = calculate_residual_loss(radial, theta, phi, energy_levels, mass, output)
        magnitude_loss = magnitude_loss_grouped(output, radial, theta, G, r_max)
        orthogonal_loss = torch.tensor(0.0, device=device) #orthogonality_loss(output, radial, theta, energy_levels, G)
        
        loss = weights[0].to(torch.float32)*residual_loss + weights[1].to(torch.float32)*magnitude_loss + weights[2].to(torch.float32)*orthogonal_loss

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()
        optim.zero_grad()
        # if i < decay_steps:
        scheduler.step()
        end = time.time()

        print(f"Step: {i+1} | Max Energy: {max_energy} | LR: {scheduler.get_last_lr()[0]} | Total Loss: {loss.item()} | Time: {1000*(end-start)}ms | Residual Loss: {residual_loss.item()} | Magnitude Loss: {magnitude_loss.item()} | Orthogonality Loss: {orthogonal_loss.item()}")
    
    
    PATH = "schrodingers_equation_hydrogen.pt"
    torch.save(model.state_dict(), PATH)