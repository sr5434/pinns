import torch
from torch import nn
import torch.nn.functional as F
import time

def sample_gamma_integer_shape(k, scale, samples, device):
    # Gamma(k, scale) for integer k via sum of exponentials.
    # Returns (samples, 1).
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")
    # (samples, k)
    u = torch.rand(samples, k, device=device).clamp_min(1e-12)
    exp_samples = -scale * torch.log(u)
    return exp_samples.sum(dim=1, keepdim=True)

def sample_radial_residual_points(n_g, l_g, samples, r_max, epsilon, device):
    k0 = 2 * int(l_g) + 3
    scale0 = torch.tensor(float(n_g) / 2.0, device=device)

    nodes = max(0, int(n_g) - int(l_g) - 1)
    if nodes <= 0:
        r = sample_gamma_integer_shape(k0, scale0, samples, device)
    else:
        # Push some samples outward to capture the outer lobes created by the Laguerre polynomial.
        # Empirically, adding ~3 per node gives a reasonable mode shift for small n.
        k1 = k0 + 3 * nodes
        s0 = samples // 2
        s1 = samples - s0
        r0 = sample_gamma_integer_shape(k0, scale0, s0, device)
        r1 = sample_gamma_integer_shape(k1, scale0, s1, device)
        r = torch.cat([r0, r1], dim=0)
        r = r[torch.randperm(samples, device=device)]

    return r.clamp_min(epsilon).clamp_max(r_max - epsilon)


def rayleigh_quotient_grouped(u, radial, l, groups):
    # Estimate E with the Rayleigh quotient for each group
    u_r = torch.autograd.grad(u, radial, grad_outputs=torch.ones_like(u), create_graph=True)[0].view(groups, -1)

    u_reshaped = u.view(groups, -1)
    radial_reshaped = radial.view(groups, -1)
    l_reshaped = l.view(groups, -1)

    numerator = torch.trapz(0.5 * (u_r ** 2) + (l_reshaped * (l_reshaped + 1) / (2 * radial_reshaped ** 2)) * (u_reshaped ** 2) - (1 / radial_reshaped) * (u_reshaped ** 2), radial_reshaped, dim=1)
    denominator = torch.trapz(u_reshaped ** 2, radial_reshaped, dim=1)
    return numerator / denominator

def calculate_residual_loss(radial, n, l, mass, u, energies):
    # Make sure that the solution is valid
    h_reduced = 1.0

    # Coulomb potential term
    V = -1.0 / radial

    u_r = torch.autograd.grad(u, radial, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    u_rr = torch.autograd.grad(u_r, radial, grad_outputs=torch.ones_like(u_r), create_graph=True)[0]
    T = -h_reduced**2/(2*mass) * u_rr + u * (h_reduced**2) * l * (l+1) / (2*mass*radial**2)
    lhs = T + V*u
    rhs = energies * u
    residual = lhs - rhs
    near_weights = torch.where(
        (l == 0),
        torch.ones_like(l, dtype=torch.float32),
        torch.full_like(l, 0.1, dtype=torch.float32),
    )
    residual_loss = torch.mean(near_weights * residual**2)
    return residual_loss

def virial_loss_grouped(u, radial, l, groups):
    radial_reshaped = radial.view(groups, -1)
    l_reshaped = l.view(groups, -1)
    u_reshaped = u.view(groups, -1)
    # Normalize u
    densities = u_reshaped**2
    densities_integral = torch.trapz(densities, radial_reshaped, dim=1).detach().reshape(groups, -1)

    u_hat_flat = u_reshaped.reshape(-1, 1)
    u_hat_r = torch.autograd.grad(
        u_hat_flat, radial,
        grad_outputs=torch.ones_like(u_hat_flat),
        create_graph=True
    )[0].view(groups, -1)

    # Coulomb potential term
    V = -1.0 / radial_reshaped
    T = 0.5 * u_hat_r**2 + l_reshaped * (l_reshaped + 1) / (2 * radial_reshaped ** 2) * u_reshaped**2

    u_hat_densities = u_reshaped**2
    T_exp = torch.trapz(T, radial_reshaped, dim=1)/densities_integral
    V_exp = torch.trapz(V * u_hat_densities, radial_reshaped, dim=1)/densities_integral

    # Per the virial theorem, <T> = -E, <V> = 2E
    # Thus, <T> + 0.5<V> = 0
    virial_term = T_exp + 0.5 * V_exp
    virial_loss = torch.mean(virial_term**2)

    return virial_loss

def magnitude_loss_grouped(u, r_base, groups):
    # u: (N,1); r_base: (P,1) shared across groups; N = groups * P
    # Spherical harmonics are already normalized so we just need to normalize the radial part
    N = u.shape[0]
    P = N // groups
    u2 = u.pow(2).view(groups, P)
    r_1d = r_base.view(P)
    integral = torch.trapezoid(u2, r_1d, dim=1)
    target = torch.ones_like(integral)
    return F.mse_loss(integral, target)

def orthogonality_loss_grouped(u, r_base, n, l, m, groups):
    N = u.shape[0]
    P = N // groups
    u_reshaped = u.view(groups, P)
    r_1d = r_base.view(P)
    overlaps = []
    for i in range(groups):
        for j in range(i+1, groups):
            # Only enforce orthogonality for states with the same (l, m) but different n
            if (l[i*P] == l[j*P]) and (m[i*P] == m[j*P]) and (n[i*P] != n[j*P]):
                inner = torch.trapezoid(u_reshaped[i] * u_reshaped[j], r_1d, dim=0)
                overlaps.append(inner**2)
    if len(overlaps) == 0:
        return torch.tensor(0.0, device=u.device)
    overlaps_tensor = torch.stack(overlaps)
    return torch.mean(overlaps_tensor)

class SchrodingerEquationHydrogen(nn.Module):
    def __init__(self, L, max_n, max_l, max_m, harmonics="complex"):
        super(SchrodingerEquationHydrogen, self).__init__()
        self.radial_net = nn.Sequential(
            nn.Linear(3, 256),
            nn.Tanh(),
            nn.Linear(256, 256),
            nn.Tanh(),
            nn.Linear(256, 128),
            nn.Tanh(),
            nn.Linear(128, 1)
        )
        
        self.L = L
        # Quantum number limits, for input normalization
        self.max_n = max_n
        self.max_l = max_l+1
        self.max_m = max_m
        self.m_range = 2*max_m+1
        self.harmonics = harmonics
    
    def analytical_spherical_harmonic(self, theta, phi, l, m):
        real = torch.zeros_like(theta)
        imag = torch.zeros_like(theta)

        mask00 = (l == 0) & (m == 0)
        real = torch.where(mask00, 0.5 * (1.0 / torch.pi) ** 0.5, real)

        mask10 = (l == 1) & (m == 0)
        real = torch.where(mask10, 0.5*torch.cos(theta) * (3.0 / torch.pi) ** 0.5, real)


        mask11 = (l == 1) & (m == 1)
        real = torch.where(mask11, -0.5*(3/(2.0*torch.pi))**0.5 * torch.sin(theta) * torch.cos(phi), real)
        imag = torch.where(mask11, -0.5*(3/(2.0*torch.pi))**0.5 * torch.sin(theta) * torch.sin(phi), imag)

        mask1m1 = (l == 1) & (m == -1)
        real = torch.where(mask1m1, 0.5*(3/(2.0*torch.pi))**0.5 * torch.sin(theta) * torch.cos(-phi), real)
        imag = torch.where(mask1m1, 0.5*(3/(2.0*torch.pi))**0.5 * torch.sin(theta) * torch.sin(-phi), imag)

        return torch.cat([real, imag], dim=-1)

    def forward(self, r, theta, phi, n, l, m):
        # Normalize quantum numbers to [0, 1]
        normed_n = (n-1)/self.max_n
        normed_l = (l)/self.max_l
        radial_inputs = torch.cat([r, normed_n, normed_l], dim=-1)
        bc = (r ** (l + 1)) * torch.exp(-r / n)
        radial = self.radial_net(radial_inputs)*bc
        # Analytical spherical harmonics (complex as [real, imag])
        angular = self.analytical_spherical_harmonic(theta, phi, l, m)

        # Enforce boundary conditions
        psi_amp = radial / (r + 1e-6)
        psi = psi_amp * angular
        return radial, psi

def cartesian_cube_3d(length, device, dtype=torch.float32):
    lin = torch.linspace(int(-length/2), int(length/2), steps=int(length), device=device, dtype=dtype)
    x, y, z = torch.meshgrid(lin, lin, lin, indexing='ij')
    points = torch.stack([x.reshape(-1), y.reshape(-1), z.reshape(-1)], dim=-1)
    dV = (lin[1] - lin[0]) ** 3
    return points, dV

def cartesian_to_spherical(x, y, z):
    r = torch.sqrt(x**2 + y**2 + z**2)
    theta = torch.acos(torch.clamp(z / (r + 1e-12), -1.0, 1.0))
    phi = torch.atan2(y, x)
    return r, theta, phi

def h2_cation(model, box_size, axis, nuclei_distance, device, dtype=torch.float32):
    grid, dV = cartesian_cube_3d(box_size, device)
    nucleus_a = torch.zeros(3, device=device, dtype=dtype)
    nucleus_b = torch.zeros(3, device=device, dtype=dtype)
    if axis == "x":
        nucleus_a[0], nucleus_b[0] = -nuclei_distance/2, nuclei_distance/2
    elif axis == "y":
        nucleus_a[1], nucleus_b[1] = -nuclei_distance/2, nuclei_distance/2
    elif axis == "z":
        nucleus_a[2], nucleus_b[2] = -nuclei_distance/2, nuclei_distance/2
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    # Coordinates relative to each nucleus
    rel_a = grid - nucleus_a.unsqueeze(0)
    rel_b = grid - nucleus_b.unsqueeze(0)

    r_a, theta_a, phi_a = cartesian_to_spherical(rel_a[:, 0], rel_a[:, 1], rel_a[:, 2])
    r_b, theta_b, phi_b = cartesian_to_spherical(rel_b[:, 0], rel_b[:, 1], rel_b[:, 2])

    n = torch.ones_like(r_a, device=device).unsqueeze(-1)
    l = torch.zeros_like(r_a, device=device).unsqueeze(-1)
    m = torch.zeros_like(r_a, device=device).unsqueeze(-1)

    _, psi_a = model(r_a.unsqueeze(-1), theta_a.unsqueeze(-1), phi_a.unsqueeze(-1), n, l, m)
    _, psi_b = model(r_b.unsqueeze(-1), theta_b.unsqueeze(-1), phi_b.unsqueeze(-1), n, l, m)

    # Overlap integral
    S = torch.sum(psi_a * psi_b) * dV

    bonding = (psi_a + psi_b) / torch.sqrt(2 * (1 + S))
    antibonding = (psi_a - psi_b) / torch.sqrt(2 * (1 - S))
    return bonding, antibonding


if __name__ == "__main__":
    device = "mps"
    # Define the model
    examples = 200000000
    steps = 20000
    length = 1.0
    mass = 1.0
    batch_size = examples//steps
    max_energy = 3
    max_sublevels = min(max_energy, 4)-1
    max_orbital = max_energy - 1
    print(f"Max Energy: {max_energy}, Max sublevels: {max_sublevels}, Max orbitals: {max_orbital}")
    r_max = 30.0
    theta_max = torch.pi
    phi_max = 2*torch.pi
    group_size = 2000 # Higher number of samples per group -> better magnitude loss approximation
    epochs = 1
    lr = 1e-3
    betas = (0.9, 0.999)
    model = SchrodingerEquationHydrogen(length, max_energy, max_sublevels, max_orbital).to(device)
    optim = torch.optim.Adam(model.parameters(), lr, betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, eta_min=1e-5, T_max=steps)

    for i in range(steps):
        start = time.time()
        weights = torch.tensor([1.0, 10.0, 1.0, 1.0], device=device) # Residual, Magnitude, Orthogonality, Virial
        # Extract batch and ensure gradients are tracked
        # Build a batch: G groups, P x-samples each
        G = batch_size//group_size
        P = batch_size//G
        pairs = torch.tensor([
            [1, 0, 0],
            [2, 0, 0],
            [2, 1, -1],
            [2, 1,  0],
            [2, 1,  1],
        ], device=device)

        idx = torch.arange(len(pairs), device=device)
        pairs_sampled = pairs[idx]

        energy_levels = pairs_sampled[:, 0:1]
        sublevel = pairs_sampled[:, 1:2]
        orbitals = pairs_sampled[:, 2:3]

        energy_levels = energy_levels.repeat_interleave(P, dim=0)
        sublevel = sublevel.repeat_interleave(P, dim=0)
        orbitals = orbitals.repeat_interleave(P, dim=0)

        epsilon = 1e-3 # to avoid zero radius/theta

        # Residual points: sample r ~ Gamma(2l+3, scale=n/2) per state.
        # This roughly matches the expected support of u(r)^2 and prevents the model from
        # "hiding" probability mass in the far tail where the Coulomb terms are tiny.
        radial_blocks = []
        theta_blocks = []
        phi_blocks = []
        for g in range(G):
            n_g = float(pairs_sampled[g, 0].item())
            l_g = int(pairs_sampled[g, 1].item())
            r_g = sample_radial_residual_points(n_g, l_g, P, r_max, epsilon, device)
            radial_blocks.append(r_g)

            mu = 2.0 * torch.rand(P, 1, device=device) - 1.0
            theta_blocks.append(torch.acos(mu))
            phi_blocks.append(torch.rand(P, 1, device=device) * phi_max)

        radial = torch.cat(radial_blocks, dim=0)
        theta = torch.cat(theta_blocks, dim=0)
        phi = torch.cat(phi_blocks, dim=0)

        radial.requires_grad_(True)
        theta.requires_grad_(True)
        phi.requires_grad_(True)

        # Deterministic (stratified) r-grid for magnitude/orthogonality/Rayleigh quotient.
        # Needed for trapezoidal integration.
        radial_norm_base = torch.linspace(epsilon, r_max - epsilon, P, device=device).unsqueeze(-1)
        radial_norm = radial_norm_base.repeat(G, 1)
        radial_norm.requires_grad_(True)
        theta_norm = torch.zeros_like(radial_norm)
        phi_norm = torch.zeros_like(radial_norm)

        # Emphasize near-origin residuals because that is where most of the "action" happens
        radial_near_max = min(3.0, r_max)
        radial_near = (radial_near_max - epsilon)*torch.rand(G*P, 1, device=device) + epsilon
        radial_near.requires_grad_(True)

        radial_output, output = model(radial, theta, phi, energy_levels, sublevel, orbitals)
        radial_output_near, output_near = model(radial_near, theta, phi, energy_levels, sublevel, orbitals)
        radial_output_norm, _ = model(radial_norm, theta_norm, phi_norm, energy_levels, sublevel, orbitals)

        rq_energies = rayleigh_quotient_grouped(radial_output_norm, radial_norm, sublevel, G)
        energies = rq_energies.detach().repeat_interleave(P).unsqueeze(-1)
        n_group = energy_levels.view(G, P)[:, 0]
        analytic_energies = -1.0 / (2.0 * (n_group ** 2))
        energy_mse = F.mse_loss(rq_energies, analytic_energies)

        residual_loss_uniform = calculate_residual_loss(
            radial, energy_levels, sublevel, mass, radial_output, energies
        )
        # Prevent the l=1 states from being dominated by near-origin residual points (which can "over-sharpen"
        # the solution and shift the donut radius). Keep full near-origin pressure for s-states.
        
        residual_loss_near = calculate_residual_loss(radial_near, energy_levels, sublevel, mass, radial_output_near, energies)
        
        residual_loss = residual_loss_uniform + residual_loss_near
        
        magnitude_loss = magnitude_loss_grouped(radial_output_norm, radial_norm_base, G)
        virial_loss = virial_loss_grouped(radial_output_norm, radial_norm, sublevel, G) # We need radial_norm instead of radial_norm_base to calculate u_r
        orthogonality_loss = orthogonality_loss_grouped(radial_output_norm, radial_norm_base, energy_levels, sublevel, orbitals, G)
        loss = weights[0].to(torch.float32)*residual_loss + weights[1].to(torch.float32)*magnitude_loss + weights[2].to(torch.float32)*orthogonality_loss + weights[3].to(torch.float32)*virial_loss
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()
        optim.zero_grad()
        scheduler.step()
        end = time.time()

        print(f"Step: {i+1} | Max Energy: {max_energy} | LR: {scheduler.get_last_lr()[0]} | Total Loss: {loss.item()} | Time: {1000*(end-start)}ms | Residual Loss: {residual_loss.item()} | Magnitude Loss: {magnitude_loss.item()} | Orthogonality Loss: {orthogonality_loss.item()} | Virial Loss: {virial_loss.item()} | Energy MSE: {energy_mse.item()}")
    
    
    PATH = "schrodingers_equation_hydrogen.pt"
    torch.save(model.state_dict(), PATH)
