import argparse
import json
import pathlib
import re
import time

import torch
import torch.nn.functional as F
from torch import nn
from torch.quasirandom import SobolEngine

# Set seeds to zero for reproducibility
torch.manual_seed(0)

# TODO: Train models for these points: 0.4, 0.8, 1.0, 1.2, 1.3, 1.4, 1.5, 1.6, 1.8, 2.2, 3.0, 5.0, 6.0

# BO total energies (electronic + nuclear repulsion) for H2 from:
# J. S. Sims and S. A. Hagstrom, J. Chem. Phys. 124, 094101 (2006), Table VI.
# https://www.nist.gov/system/files/documents/2017/05/09/sims-h2.pdf
_H2_BO_REFERENCE_ENERGIES = {
    "0.4": -0.1202303411778644,
    "0.5": -0.5266387587423172,
    "0.6": -0.7696354294853568,
    "0.7": -0.9220274615270152,
    "0.8": -1.0200566663601389,
    "0.9": -1.0836432399585087,
    "1": -1.1245397195465791,
    "1.1": -1.1500573677382885,
    "1.2": -1.1649352434400281,
    "1.3": -1.17234714903778,
    "1.4": -1.1744757142200755,
    "1.5": -1.1728550795781447,
    "1.6": -1.1685833733709263,
    "1.7": -1.1624587268978088,
    "1.8": -1.1550687376108071,
    "1.9": -1.146850697028721,
    "2": -1.1381329571315035,
    "2.1": -1.1291638360999721,
    "2.2": -1.1201321168476391,
    "2.3": -1.1111817652026448,
    "2.4": -1.1024226060092978,
    "2.5": -1.0939381299535998,
    "2.6": -1.0857912373935887,
    "2.7": -1.0780284841810479,
    "2.8": -1.0706832334784095,
    "2.9": -1.0637780088027916,
    "3": -1.0573262688692439,
    "3.1": -1.0513337722644516,
    "3.2": -1.0457996614287338,
    "3.3": -1.0407173653475985,
    "3.4": -1.0360753951869195,
    "3.5": -1.031858084851223,
    "3.6": -1.0280463083758766,
    "3.7": -1.0246181884071472,
    "3.8": -1.0215497955299109,
    "3.9": -1.0188158276928498,
    "4": -1.0163902529471283,
    "4.2": -1.0123599596799189,
    "4.4": -1.0092565162586632,
    "4.6": -1.0068952238201211,
    "4.8": -1.0051160060980952,
    "5": -1.0037856585819889,
    "5.2": -1.0027968163095431,
    "5.4": -1.0020650572082353,
    "5.6": -1.0015252518853549,
    "5.8": -1.0011278808513214,
    "6": -1.0008357076542279,
}
H2_BONDING_AUDIT_BOX_LENGTH = 8.0
H2_BONDING_AUDIT_BATCH_SIZE = 9_375
H2_BONDING_AUDIT_REPEATS = 16
H2_MODEL_DIR = pathlib.Path("h2_models")
H2_FULL_CHECKPOINT_PREFIX = "schrodingers_equation_h2_ground"
H2_ENERGY_NORM_ONLY_CHECKPOINT_PREFIX = "energy_norm_only_schrodingers_equation_h2_ground"
H2_CHECKPOINT_PATTERN_TEMPLATE = r"{prefix}_R(?P<R>[0-9]+(?:\.[0-9]+)?)\.pt$"
H2_CHECKPOINT_PATTERN = re.compile(
    H2_CHECKPOINT_PATTERN_TEMPLATE.format(prefix=re.escape(H2_FULL_CHECKPOINT_PREFIX))
)


def get_device(device_name="auto"):
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return "mps"
        if torch.cuda.is_available():
            return "cuda"
        return "cpu"
    if device_name == "same":
        raise ValueError("'same' is only valid for eval-device, not the training device.")
    return device_name


def format_h2_distance_tag(nuclei_distance, decimals=6):
    text = format(float(nuclei_distance), f".{int(decimals)}f").rstrip("0").rstrip(".")
    return text or "0"


def h2_checkpoint_prefix(energy_norm_only=False):
    if energy_norm_only:
        return H2_ENERGY_NORM_ONLY_CHECKPOINT_PREFIX
    return H2_FULL_CHECKPOINT_PREFIX


def default_h2_checkpoint_path(nuclei_distance, energy_norm_only=False):
    distance_tag = format_h2_distance_tag(nuclei_distance)
    return H2_MODEL_DIR / f"{h2_checkpoint_prefix(energy_norm_only)}_R{distance_tag}.pt"


def get_existing_h2_checkpoint_path(nuclei_distance, energy_norm_only=False):
    preferred = default_h2_checkpoint_path(
        nuclei_distance, energy_norm_only=energy_norm_only
    )
    if energy_norm_only:
        return preferred
    legacy_r_tagged = pathlib.Path(f"schrodingers_equation_h2_ground_R{format_h2_distance_tag(nuclei_distance)}.pt")
    legacy = pathlib.Path("schrodingers_equation_h2_ground.pt")
    if preferred.exists():
        return preferred
    if legacy_r_tagged.exists():
        return legacy_r_tagged
    if legacy.exists():
        return legacy
    return preferred


def discover_h2_checkpoints(models_dir=H2_MODEL_DIR, energy_norm_only=False):
    models_dir = pathlib.Path(models_dir)
    matches = {}
    if not models_dir.exists():
        return []

    prefix = h2_checkpoint_prefix(energy_norm_only)
    checkpoint_pattern = re.compile(
        H2_CHECKPOINT_PATTERN_TEMPLATE.format(prefix=re.escape(prefix))
    )
    for path in sorted(models_dir.glob(f"{prefix}_R*.pt")):
        match = checkpoint_pattern.fullmatch(path.name)
        if match is None:
            continue
        nuclei_distance = float(match.group("R"))
        matches.setdefault(format_h2_distance_tag(nuclei_distance), (nuclei_distance, path))
    return sorted(matches.values(), key=lambda item: item[0])


def lookup_h2_reference_total_energy(nuclei_distance):
    distance_tag = format_h2_distance_tag(nuclei_distance)
    energy = _H2_BO_REFERENCE_ENERGIES.get(distance_tag)
    if energy is None:
        supported = ", ".join(_H2_BO_REFERENCE_ENERGIES.keys())
        raise ValueError(
            f"No tabulated H2 reference energy for R={float(nuclei_distance):g}. "
            f"Supported R values: {supported}"
        )
    return {
        "energy_hartree": float(energy),
        "R_reference": float(distance_tag),
    }

def build_cartesian_grid_6d(max_points, length, device, dtype=torch.float32):
    points_per_dim = max(2, int(round(max_points ** (1.0 / 6.0))))
    while (points_per_dim + 1) ** 6 <= max_points:
        points_per_dim += 1
    while points_per_dim > 2 and points_per_dim ** 6 > max_points:
        points_per_dim -= 1

    lin = torch.linspace(-length / 2, length / 2, steps=points_per_dim, device=device, dtype=dtype)
    axes = [lin] * 6
    # Meshgrid is needed for trapezoidal integration
    mesh = torch.meshgrid(*axes, indexing="ij")
    return torch.stack([m.reshape(-1) for m in mesh], dim=-1)


def build_sobol_points_6d(num_points, length, sobol_engine, device, dtype=torch.float32):
    # QMC points in [0,1)^6 mapped into [-L/2, L/2]^6.
    points_01 = sobol_engine.draw(num_points, dtype=dtype)
    return (length * points_01 - (length / 2)).to(device=device, dtype=dtype)


def _integrate_trapezoid_tensor_product(values, coordinates):
    # Uses trapezoidal integration only when points form a full tensor-product grid.
    flat_values = values.reshape(-1)
    detached_coords = coordinates.detach()
    N, dims = detached_coords.shape

    unique_per_dim = []
    inverse_per_dim = []
    sizes = []
    total_points = 1
    for dim in range(dims):
        unique_dim, inverse_dim = torch.unique(
            detached_coords[:, dim], sorted=True, return_inverse=True
        )
        size = unique_dim.numel()
        if size < 2:
            return None
        unique_per_dim.append(unique_dim)
        inverse_per_dim.append(inverse_dim)
        sizes.append(size)
        total_points *= int(size)

    if total_points != N:
        return None

    flat_indices = torch.zeros(N, device=coordinates.device, dtype=torch.long)
    stride = 1
    for dim in range(dims - 1, -1, -1):
        flat_indices += inverse_per_dim[dim] * stride
        stride *= sizes[dim]

    sorted_idx = torch.argsort(flat_indices)
    expected = torch.arange(N, device=coordinates.device, dtype=torch.long)
    if not torch.equal(flat_indices[sorted_idx], expected):
        return None

    tensor_values = flat_values[sorted_idx].reshape(*sizes)
    for dim in range(dims - 1, -1, -1):
        tensor_values = torch.trapezoid(tensor_values, unique_per_dim[dim], dim=dim)
    return tensor_values


def integrate_6d(values, coordinates, eps=1e-12):
    trapz_integral = _integrate_trapezoid_tensor_product(values, coordinates)
    if trapz_integral is not None:
        return trapz_integral

    detached_coords = coordinates.detach()
    mins = torch.amin(detached_coords, dim=0)
    maxs = torch.amax(detached_coords, dim=0)
    volume = torch.prod((maxs - mins).clamp_min(eps))
    return torch.mean(values.reshape(-1)) * volume


def integrate_box_qmc(values, box_length, dims=6):
    volume = values.new_tensor(float(box_length) ** int(dims))
    return torch.mean(values.reshape(-1)) * volume


def calculate_rayleigh_quotient(hamiltonian_psi, psi, coordinates=None, box_length=None, eps=1e-8):
    # Rayleigh quotient for the electronic Hamiltonian: <psi|H|psi>/<psi|psi>.
    if box_length is not None:
        numerator = integrate_box_qmc(psi * hamiltonian_psi, box_length=box_length)
        denominator = integrate_box_qmc(psi.pow(2), box_length=box_length)
    elif coordinates is not None:
        numerator = integrate_6d(psi * hamiltonian_psi, coordinates)
        denominator = integrate_6d(psi.pow(2), coordinates)
    else:
        raise ValueError("Either coordinates or box_length must be provided.")
    denominator = denominator.clamp_min(eps)
    return numerator / denominator

def calculate_hamiltonian(psi, logpsi, coordinates, r1A, r1B, r2A, r2B, r12):
    # Electronic Born-Oppenheimer potential (nuclear repulsion handled outside H_elec).
    V = -1.0 / r1A - 1.0 / r1B - 1.0 / r2A - 1.0 / r2B + 1.0 / r12

    grad_logpsi = torch.autograd.grad(
        logpsi, coordinates, grad_outputs=torch.ones_like(logpsi), create_graph=True
    )[0]  # (N,6)

    lap_logpsi = torch.zeros_like(logpsi)  # (N,1)
    for d in range(coordinates.shape[1]):
        g = grad_logpsi[:, d:d+1]
        lap_logpsi += torch.autograd.grad(
            g, coordinates, grad_outputs=torch.ones_like(g), create_graph=True
        )[0][:, d:d+1]

    T_local = -0.5 * (lap_logpsi + (grad_logpsi**2).sum(dim=1, keepdim=True))
    kinetic = psi * T_local
    hamiltonian = kinetic + V * psi


    return hamiltonian, kinetic, V


def calculate_residual_loss(hamiltonian, psi, energy, coordinates=None, box_length=None, eps=1e-8):
    residual_sq = (hamiltonian - psi * energy).pow(2)
    return residual_sq.mean()


def calculate_magnitude_loss(psi, coordinates=None, box_length=None):
    if box_length is not None:
        norm = integrate_box_qmc(psi.pow(2), box_length=box_length)
    elif coordinates is not None:
        norm = integrate_6d(psi.pow(2), coordinates)
    else:
        raise ValueError("Either coordinates or box_length must be provided.")
    target = torch.tensor(1.0, device=psi.device, dtype=psi.dtype)
    return F.mse_loss(norm, target), norm


def calculate_virial_loss(kinetic, potential, psi, coordinates=None, box_length=None, eps=1e-8):
    density = psi.pow(2)
    if box_length is not None:
        norm = integrate_box_qmc(density, box_length=box_length).clamp_min(eps)
        kinetic_exp = integrate_box_qmc(psi * kinetic, box_length=box_length) / norm
        potential_exp = integrate_box_qmc(potential * density, box_length=box_length) / norm
    elif coordinates is not None:
        norm = integrate_6d(density, coordinates).clamp_min(eps)
        kinetic_exp = integrate_6d(psi * kinetic, coordinates) / norm
        potential_exp = integrate_6d(potential * density, coordinates) / norm
    else:
        raise ValueError("Either coordinates or box_length must be provided.")
    virial_term = 2.0 * kinetic_exp + potential_exp
    return virial_term.pow(2)


def calculate_variance_loss(hamiltonian, psi, energy, coordinates=None, box_length=None, eps=1e-8):
    # Numerically stable form: Var[E_L] = <(Hpsi - Epsi)^2> / <psi^2>.
    
    residual_sq = (hamiltonian / psi - energy).pow(2) * psi**2
    if box_length is not None:
        numerator = integrate_box_qmc(residual_sq, box_length=box_length)
        denominator = integrate_box_qmc(psi.pow(2), box_length=box_length).clamp_min(eps)
    elif coordinates is not None:
        numerator = integrate_6d(residual_sq, coordinates)
        denominator = integrate_6d(psi.pow(2), coordinates).clamp_min(eps)
    else:
        raise ValueError("Either coordinates or box_length must be provided.")
    return numerator / denominator


@torch.no_grad()
def estimate_norm_qmc(model, num_points, box_length, sobol_engine, nuclei_distance, axis, device, chunk_size=32768):
    if num_points <= 0:
        raise ValueError("num_points must be > 0")

    total_density = torch.tensor(0.0, device=device)
    total_count = 0
    while total_count < num_points:
        n = min(chunk_size, num_points - total_count)
        points = build_sobol_points_6d(n, box_length, sobol_engine, device=device)
        r1A, r1B, r2A, r2B, r12 = model.pair_distances_from_collocation_points(
            points, nuclei_distance=nuclei_distance, axis=axis
        )
        psi, _ = model(r1A, r1B, r2A, r2B, r12)
        total_density += psi.pow(2).sum()
        total_count += n

    mean_density = total_density / float(total_count)
    volume = float(box_length) ** 6
    return mean_density * volume

def compute_quadrupole(points, psi, nuclei_distance, axis="x"):
    coords = {
        "x": (points[:,0], points[:,3]),
        "y": (points[:,1], points[:,4]),
        "z": (points[:,2], points[:,5]),
    }
    a1, a2 = coords[axis]

    r1_sq = points[:,0]**2 + points[:,1]**2 + points[:,2]**2
    r2_sq = points[:,3]**2 + points[:,4]**2 + points[:,5]**2

    Q_elec = (3*a1**2 - r1_sq) + (3*a2**2 - r2_sq)
    density = psi.reshape(-1).pow(2)
    numerator = torch.mean(Q_elec * density)
    denominator = torch.mean(density)

    Q_expect = numerator / denominator

    # nuclei contribution
    R = nuclei_distance
    Q_nuc = R**2

    return -Q_expect + Q_nuc

def evaluate_h2_checkpoint(
    checkpoint_path,
    nuclei_distance,
    axis="x",
    device="cpu",
    box_length=H2_BONDING_AUDIT_BOX_LENGTH,
    batch_size=H2_BONDING_AUDIT_BATCH_SIZE,
    n_repeats=H2_BONDING_AUDIT_REPEATS,
):
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if n_repeats <= 0:
        raise ValueError("n_repeats must be > 0")

    checkpoint_path = pathlib.Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"H2 checkpoint not found: {checkpoint_path}")

    reference = lookup_h2_reference_total_energy(nuclei_distance)
    device = torch.device(device)
    model = SchrodingerEquationH2Ground().to(device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    energies = []
    quadrupoles = []
    rayleigh_gaps = []
    with torch.enable_grad():
        for seed in range(int(n_repeats)):
            rq_pair = []
            quadrupole_pair = []
            for sobol_seed in (100 + seed, 500 + seed):
                sobol_engine = SobolEngine(dimension=6, scramble=True, seed=sobol_seed)
                points = build_sobol_points_6d(
                    batch_size, box_length, sobol_engine, device=device
                ).requires_grad_(True)
                r1A, r1B, r2A, r2B, r12 = model.pair_distances_from_collocation_points(
                    points, nuclei_distance=nuclei_distance, axis=axis
                )
                psi, logpsi = model(r1A, r1B, r2A, r2B, r12)
                hamiltonian, _, _ = calculate_hamiltonian(
                    psi, logpsi, points, r1A, r1B, r2A, r2B, r12
                )
                rq_pair.append(
                    float(
                        calculate_rayleigh_quotient(
                            hamiltonian, psi, box_length=box_length
                        ).detach().cpu()
                    )
                )
                quadrupole_pair.append(
                    float(
                        compute_quadrupole(
                            points, psi, nuclei_distance, axis=axis
                        ).detach().cpu()
                    )
                )

            rq_energy_a, rq_energy_b = rq_pair
            quadrupole_a, quadrupole_b = quadrupole_pair
            energies.append((rq_energy_a + rq_energy_b) / 2.0 + 1.0 / float(nuclei_distance))
            quadrupoles.append((quadrupole_a+quadrupole_b) / 2.0)
            rayleigh_gaps.append(abs(rq_energy_a - rq_energy_b))

    energies_t = torch.tensor(energies, dtype=torch.float64)
    quadrupoles_t = torch.tensor(quadrupoles, dtype=torch.float64)
    reference_energy = float(reference["energy_hartree"])
    return energies_t.mean().item(), abs(reference_energy - energies_t.mean().item()), quadrupoles_t.mean().item()


def calculate_rayleigh_consistency_loss(rayleigh_energy_a, rayleigh_energy_b, rayleigh_energy_norm):
    # Prevent optimizer from exploiting one particular sample set.
    return (rayleigh_energy_a - rayleigh_energy_b).pow(2) + (rayleigh_energy_a - rayleigh_energy_norm).pow(2) + (rayleigh_energy_b - rayleigh_energy_norm).pow(2)


def calculate_rayleigh_advantage_loss(rayleigh_energy, energy_baseline, max_negative_advantage=0.05):
    # Optimize energy relative to EMA baseline so absolute negative scale does not dominate total loss.
    raw_advantage = rayleigh_energy - energy_baseline.detach()
    # Sort of like PPO clipping
    # Our estimator is noisy, so we do not want to be too trusting
    clipped_advantage = raw_advantage.clamp(min=-max_negative_advantage)
    return clipped_advantage, raw_advantage


def calculate_rayleigh_reliability_gate(
    rayleigh_energy_a,
    rayleigh_energy_b,
    norm_train,
    virial_loss,
    rayleigh_consistency_loss,
    k_gap=6.0,
    k_norm=3.0,
    k_virial=0.75,
    k_consistency=2.0,
    min_gate=0.0,
    max_gate=1.0,
):
    # Downweight direct energy minimization when reliability signals are weak/noisy.
    rayleigh_gap = torch.abs(rayleigh_energy_a - rayleigh_energy_b).detach()
    norm_error = torch.abs(norm_train - norm_train.new_tensor(1.0)).detach()
    virial_term = virial_loss.detach()
    consistency_term = torch.sqrt(rayleigh_consistency_loss.detach().clamp_min(0.0))
    gate = (
        torch.exp(-k_gap * rayleigh_gap)
        * torch.exp(-k_norm * norm_error)
        * torch.exp(-k_virial * virial_term)
        * torch.exp(-k_consistency * consistency_term)
    )
    return gate.clamp(min=min_gate, max=max_gate), rayleigh_gap, norm_error


def get_training_loss_weights(step, total_steps, device, dtype=torch.float32, rayleigh_weight_coefficient=68.0):
    """
    Phase A: Let PDE.
    Phase B: introduce small Rayleigh pressure with stronger consistency/variance.
    Phase C: mild Rayleigh refinement once constraints are more stable.
    """
    phase_a_end = max(1, int(0.30 * total_steps))
    phase_b_end = max(phase_a_end + 1, int(0.75 * total_steps))

    if step < phase_a_end:
        # Let PDE residual and normalization dominate to let training stabilize
        # Do not minimize energy at all
        # residual, magnitude, orthogonality, virial, variance, consistency, rayleigh
        vals = [1.0, 10.0, 0.0, 0.1, 2.0, 0.4, 0.0]
        return torch.tensor(vals, device=device, dtype=dtype)

    if step < phase_b_end:
        # Add energy minimization(linearly warm up weight), weaken variance to give model more wiggle room
        t = float(step - phase_a_end) / float(max(1, phase_b_end - phase_a_end))
        rayleigh_w = 0.1 + 0.5 * t
        # R=0.4: 850
        # R=0.8: 160.21
        # R=1.0: 85
        # R=1.1: 63
        # R=1.2: 49.1 # (Increase weight)
        # R=1.3: 40.8 # (Increase weight very slightly)
        # R=1.4: 35
        # R=1.5: 29.2 (Decrease weight)
        # R=1.6: 22 (Decrease weight)
        # R=1.7: 21.5
        # R=1.8: 19
        # R=2.2: 13.56
        # R=3.0: 7.45
        # R=5.0: 3.22
        # R=6.0: 2.9
        vals = [1.0, 10.0, 0.0, 1.0, 4.0, 0.6, rayleigh_weight_coefficient * rayleigh_w]
        return torch.tensor(vals, device=device, dtype=dtype)

    # Warm up energy weight at a steeper rate, weaken some constraints on energy (variance, and consistency) while strengthening virial
    t = float(step - phase_b_end) / float(max(1, total_steps - phase_b_end))
    rayleigh_w = 0.6 + 0.4 * t
    vals = [1.0, 8.0, 0.0, 4.0, 7.0, 0.8, rayleigh_weight_coefficient * rayleigh_w]
    return torch.tensor(vals, device=device, dtype=dtype)


def get_energy_norm_only_loss_weights(device, dtype=torch.float32, rayleigh_weight_coefficient=68.0):
    # residual, magnitude, orthogonality, virial, variance, consistency, rayleigh
    vals = [0.0, 10.0, 0.0, 0.0, 0.0, 0.0, rayleigh_weight_coefficient]
    return torch.tensor(vals, device=device, dtype=dtype)


class SchrodingerEquationH2Ground(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        self.core = nn.Sequential(
            nn.Linear(4, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden), nn.Tanh(),
            nn.Linear(hidden, hidden // 2), nn.Tanh(),
            nn.Linear(hidden // 2, 1),
        )

        self.cA = nn.Parameter(torch.tensor(0.01))
        self.cB = nn.Parameter(torch.tensor(0.01))
        # Raw parameter; transformed to positive b in forward for a pole-free Jastrow denominator.
        b_init = torch.tensor(0.5)
        
        b_raw_init = torch.log(torch.expm1(b_init)) # We initialize like this so softplus(b)=0.5
        self.b = nn.Parameter(b_raw_init)

        alpha_delta_init = torch.tensor(0.0)
        self.alpha_delta = nn.Parameter(alpha_delta_init)
        # Sigmoid of this parameter is the weight of the ionic term in the LCAO ansatz.
        # Start with a mild suppression of the ionic component.
        self.ionic_raw = nn.Parameter(torch.tensor(1.5))


    def _features(self, r1A, r1B, r2A, r2B, r12):
        # Symmetric combinations keep psi(r1,r2)=psi(r2,r1) easier to enforce
        s1 = r1A + r2A
        s2 = r1B + r2B
        s = s1+s2
        d = torch.abs(s1-s2)
        return torch.cat([s, d, r12, s1*s2], dim=-1)

    def singlet_spin_factor(self, spin_e1, spin_e2):
        if spin_e1 == "up" and spin_e2 == "down":
            return 1.0/torch.sqrt(torch.tensor(2.0))
        if spin_e1 == "down" and spin_e2 == "up":
            return -1.0/torch.sqrt(torch.tensor(2.0))
        return 0.0 # Pauli exclusion for same-spin electrons

    def forward(self, r1A, r1B, r2A, r2B, r12, spin_e1="up", spin_e2="down", account_for_spin=False):
        log_base = self.core(self._features(r1A, r1B, r2A, r2B, r12))
        # Prevent sign changes, ensuring there are no nodes
        base = torch.exp(log_base)
        # Jastrow factor enforces electron-electron cusp
        b_eff = F.softplus(self.b)
        two_body = 0.5 * r12 / (1.0 + b_eff * r12)
        three_body = self.cA * (r1A+r2A) * r12 + self.cB * (r1B+r2B) * r12
        jastrow = torch.exp(two_body + three_body)
        # LCAO enforces bonding symmetry
        alpha = 1.0 + 0.1 * torch.tanh(self.alpha_delta)

        phiA1 = torch.exp(-alpha * r1A)
        phiB1 = torch.exp(-alpha * r1B)
        phiA2 = torch.exp(-alpha * r2A)
        phiB2 = torch.exp(-alpha * r2B)

        # Construct a linear combination of covalent and ionic terms
        covalent = phiA1 * phiB2 + phiB1 * phiA2
        ionic = phiA1 * phiA2 + phiB1 * phiB2
        ionic_weight = torch.sigmoid(self.ionic_raw)
        lcao_backbone = covalent + ionic_weight * ionic
        log_lcao_backbone = torch.log(lcao_backbone.clamp_min(1e-12))

        # If enabled, calculate spin factor for singlet state.
        # Do not calculate spin during training
        if account_for_spin and not self.training:
            spin_factor = self.singlet_spin_factor(spin_e1, spin_e2)
        else:
            spin_factor = 1.0
        # Because the spin factor is made negative for one of the two possible opposite-spin configurations, it is ignored when calculating logpsi
        return base * jastrow * lcao_backbone * spin_factor, (log_base + two_body + three_body + log_lcao_backbone)

    @staticmethod
    def pair_distances_from_collocation_points(collocation_points, nuclei_distance, axis="x", eps=1e-8):
        if collocation_points.ndim == 2 and collocation_points.shape[1] == 6:
            points = collocation_points.view(-1, 2, 3)
        elif collocation_points.ndim == 3 and collocation_points.shape[1:] == (2, 3):
            points = collocation_points
        else:
            raise ValueError(
                "collocation_points must have shape (N, 6) or (N, 2, 3)"
            )

        axis_to_dim = {"x": 0, "y": 1, "z": 2}
        if axis not in axis_to_dim:
            raise ValueError("axis must be 'x', 'y', or 'z'")

        device = collocation_points.device
        dtype = collocation_points.dtype
        nucleus_a = torch.zeros(3, device=device, dtype=dtype)
        nucleus_b = torch.zeros(3, device=device, dtype=dtype)
        axis_idx = axis_to_dim[axis]
        half_distance = 0.5 * nuclei_distance
        nucleus_a[axis_idx] = -half_distance
        nucleus_b[axis_idx] = half_distance

        electron_1 = points[:, 0, :]
        electron_2 = points[:, 1, :]

        def distance(a, b):
            diff = a - b
            # We need an epsilon so the potential energy does not blow up
            return torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True))

        r1A = distance(electron_1, nucleus_a).clamp_min(eps)
        r1B = distance(electron_1, nucleus_b).clamp_min(eps)
        r2A = distance(electron_2, nucleus_a).clamp_min(eps)
        r2B = distance(electron_2, nucleus_b).clamp_min(eps)
        r12 = distance(electron_1, electron_2).clamp_min(eps)
        return r1A, r1B, r2A, r2B, r12#_eff

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


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the H2 ground-state PINN and run the bonding-analysis repeated Sobol audit."
    )
    parser.add_argument("--R", type=float, default=1.4, help="Internuclear distance in Bohr.")
    parser.add_argument(
        "--axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis along which the nuclei are separated.",
    )
    parser.add_argument("--steps", type=int, default=20_000, help="Number of optimization steps.")
    parser.add_argument(
        "--examples",
        type=int,
        default=37_500_000 * 5,
        help="Total effective training examples; batch size is examples / steps.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=("auto", "cpu", "mps", "cuda"),
        help="Training device.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=None,
        help="Checkpoint path. Default is h2_models/schrodingers_equation_h2_ground_R{R}.pt",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the post-training bonding_analysis-style repeated Sobol audit.",
    )
    parser.add_argument(
        "--eval-device",
        default="cpu",
        choices=("cpu", "mps", "cuda", "same"),
        help="Device used by the post-training audit.",
    )
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=H2_BONDING_AUDIT_REPEATS,
        help="Number of repeated Sobol audits for the post-training evaluation.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=H2_BONDING_AUDIT_BATCH_SIZE,
        help="Sobol sample count per audit draw, matching bonding_analysis.ipynb by default.",
    )
    parser.add_argument(
        "--eval-box-length",
        type=float,
        default=H2_BONDING_AUDIT_BOX_LENGTH,
        help="Box length used during the post-training bonding-analysis audit.",
    )
    parser.add_argument(
        "--rayleigh-weight-coefficient",
        type=float,
        default=68.0,
        help="Base coefficient for the phased Rayleigh energy-minimization weight schedule.",
    )
    parser.add_argument(
        "--energy-norm-only",
        action="store_true",
        help=(
            "Train with only Rayleigh energy minimization and normalization loss. "
            "Residual, orthogonality, virial, variance, and consistency weights are zeroed, "
            "and the default checkpoint gets an energy_norm_only_ prefix."
        ),
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = get_device(args.device)

    if args.steps <= 0:
        raise ValueError("--steps must be > 0")
    if args.examples < args.steps:
        raise ValueError("--examples must be >= --steps so the training batch size stays positive")
    if args.rayleigh_weight_coefficient < 0:
        raise ValueError("--rayleigh-weight-coefficient must be >= 0")

    examples = int(args.examples)
    steps = int(args.steps)
    batch_size = examples // steps
    batch_size_residual = max(1, batch_size // 2)
    batch_size_residual_near = max(1, batch_size // 4)
    R = float(args.R)
    axis = args.axis
    checkpoint_path = (
        pathlib.Path(args.output)
        if args.output is not None
        else default_h2_checkpoint_path(R, energy_norm_only=args.energy_norm_only)
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    reference_info = lookup_h2_reference_total_energy(R)
    reference_total_energy = torch.tensor(reference_info["energy_hartree"], device=device)

    # Define the model
    L = 8  # Length of box for sampling
    L_near = 4  # Length of box for upsampling near singular regions
    lr = 1e-3
    betas = (0.9, 0.999)
    e_bar_decay = 0.98
    lambda_qmc = 0.2  # How much influence Sobol points have on the residual calculation
    norm_batch_size = min(batch_size, 8192)
    norm_audit_points = 131072
    norm_audit_every = 100
    model = SchrodingerEquationH2Ground().to(device)
    optim = torch.optim.Adam(model.parameters(), lr, betas)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, eta_min=1e-5, T_max=steps)
    sobol_engine_energy_a = SobolEngine(dimension=6, scramble=True, seed=11)
    sobol_engine_energy_b = SobolEngine(dimension=6, scramble=True, seed=53)
    sobol_engine_norm = SobolEngine(dimension=6, scramble=True, seed=29)
    sobol_engine_norm_audit = SobolEngine(dimension=6, scramble=True, seed=97)
    e_bar = None

    print(
        f"Training H2 ground state | device={device} | R={R:g} Bohr | axis={axis} | steps={steps} | "
        f"batch_size={batch_size} | rayleigh_weight_coefficient={args.rayleigh_weight_coefficient:g} "
        f"| energy_norm_only={args.energy_norm_only} "
        f"| checkpoint={checkpoint_path}"
    )

    for i in range(steps):
        start = time.time()
        # Residual, Magnitude, Orthogonality, Virial, Variance, Rayleigh consistency, energy minimization
        if args.energy_norm_only:
            weights = get_energy_norm_only_loss_weights(
                device=device,
                rayleigh_weight_coefficient=args.rayleigh_weight_coefficient,
            )
        else:
            weights = get_training_loss_weights(
                i,
                steps,
                device=device,
                rayleigh_weight_coefficient=args.rayleigh_weight_coefficient,
            )

        collocation_points = (L * torch.rand(batch_size_residual, 6, device=device) - L / 2).requires_grad_(True)
        r1A, r1B, r2A, r2B, r12 = model.pair_distances_from_collocation_points(
            collocation_points, nuclei_distance=R, axis=axis
        )

        # QMC points stabilize normalization-sensitive losses versus tiny tensor-product grids.
        norm_points = build_sobol_points_6d(norm_batch_size, L, sobol_engine_norm, device=device).requires_grad_(True)
        r1A_norm, r1B_norm, r2A_norm, r2B_norm, r12_norm = model.pair_distances_from_collocation_points(
            norm_points, nuclei_distance=R, axis=axis
        )

        # Near-region points for more stable residual around singular Coulomb regions.
        collocation_points_near = (
            L_near * torch.rand(batch_size_residual_near, 6, device=device) - L_near / 2
        ).requires_grad_(True)
        r1A_near, r1B_near, r2A_near, r2B_near, r12_near = model.pair_distances_from_collocation_points(
            collocation_points_near, nuclei_distance=R, axis=axis
        )
        # Two independent QMC energy sets prevent single-sample-set exploitation.
        energy_points_a = build_sobol_points_6d(batch_size, L, sobol_engine_energy_a, device=device).requires_grad_(True)
        r1A_energy_a, r1B_energy_a, r2A_energy_a, r2B_energy_a, r12_energy_a = model.pair_distances_from_collocation_points(
            energy_points_a, nuclei_distance=R, axis=axis
        )
        energy_points_b = build_sobol_points_6d(batch_size, L, sobol_engine_energy_b, device=device).requires_grad_(True)
        r1A_energy_b, r1B_energy_b, r2A_energy_b, r2B_energy_b, r12_energy_b = model.pair_distances_from_collocation_points(
            energy_points_b, nuclei_distance=R, axis=axis
        )

        output, log_output = model(r1A, r1B, r2A, r2B, r12)
        output_norm, log_output_norm = model(r1A_norm, r1B_norm, r2A_norm, r2B_norm, r12_norm)
        output_near, log_output_near = model(r1A_near, r1B_near, r2A_near, r2B_near, r12_near)
        output_energy_a, log_output_energy_a = model(r1A_energy_a, r1B_energy_a, r2A_energy_a, r2B_energy_a, r12_energy_a)
        output_energy_b, log_output_energy_b = model(r1A_energy_b, r1B_energy_b, r2A_energy_b, r2B_energy_b, r12_energy_b)
        hamiltonian, kinetic, potential = calculate_hamiltonian(
            output, log_output, collocation_points, r1A, r1B, r2A, r2B, r12
        )
        hamiltonian_norm, kinetic_norm, potential_norm = calculate_hamiltonian(
            output_norm, log_output_norm, norm_points, r1A_norm, r1B_norm, r2A_norm, r2B_norm, r12_norm
        )
        hamiltonian_near, kinetic_near, potential_near = calculate_hamiltonian(
            output_near, log_output_near, collocation_points_near, r1A_near, r1B_near, r2A_near, r2B_near, r12_near
        )
        hamiltonian_energy_a, _, _ = calculate_hamiltonian(
            output_energy_a, log_output_energy_a, energy_points_a, r1A_energy_a, r1B_energy_a, r2A_energy_a, r2B_energy_a, r12_energy_a
        )
        hamiltonian_energy_b, _, _ = calculate_hamiltonian(
            output_energy_b, log_output_energy_b, energy_points_b, r1A_energy_b, r1B_energy_b, r2A_energy_b, r2B_energy_b, r12_energy_b
        )

        rq_energy_a = calculate_rayleigh_quotient(hamiltonian_energy_a, output_energy_a, box_length=L)
        rq_energy_b = calculate_rayleigh_quotient(hamiltonian_energy_b, output_energy_b, box_length=L)
        rq_energy_norm = calculate_rayleigh_quotient(hamiltonian_norm, output_norm, box_length=L)
        rq_energy = (rq_energy_a + rq_energy_b + rq_energy_norm) / 3.0
        # Use EMA of Rayleigh quotient as a more stable target for residual/variance losses
        rq_energy_detached = rq_energy.detach()
        if e_bar is None:
            e_bar = rq_energy_detached.clone()
        else:
            e_bar = e_bar_decay * e_bar + (1.0 - e_bar_decay) * rq_energy_detached
        rayleigh_consistency_loss = calculate_rayleigh_consistency_loss(rq_energy_a, rq_energy_b, rq_energy_norm)
        rq_energy_total = rq_energy + (1.0 / R)  # Add nuclear repulsion after the electronic Rayleigh quotient
        energy_mae = torch.mean(torch.abs(rq_energy_total - reference_total_energy))
        residual_loss_uniform = calculate_residual_loss(hamiltonian, output, e_bar, box_length=L)
        residual_loss_near = calculate_residual_loss(hamiltonian_near, output_near, e_bar, box_length=L_near)
        residual_loss_norm = calculate_residual_loss(hamiltonian_norm, output_norm, e_bar, box_length=L)
        residual_loss_energy_a = calculate_residual_loss(
            hamiltonian_energy_a, output_energy_a, e_bar, box_length=L
        )
        residual_loss_energy_b = calculate_residual_loss(
            hamiltonian_energy_b, output_energy_b, e_bar, box_length=L
        )
        residual_loss_energy = 0.5 * (residual_loss_energy_a + residual_loss_energy_b)
        residual_loss = (
            residual_loss_uniform
            + residual_loss_near
            + lambda_qmc * (residual_loss_norm + residual_loss_energy)
        )

        magnitude_loss, norm_train = calculate_magnitude_loss(output_norm, box_length=L)
        virial_loss = calculate_virial_loss(kinetic_norm, potential_norm, output_norm, box_length=L)
        variance_loss = calculate_variance_loss(hamiltonian_norm, output_norm, rq_energy_norm.detach(), box_length=L)
        if args.energy_norm_only:
            rayleigh_gap = torch.abs(rq_energy_a - rq_energy_b).detach()
            norm_error = torch.abs(norm_train - norm_train.new_tensor(1.0)).detach()
            rayleigh_gate = torch.ones((), device=device, dtype=rq_energy.dtype)
        else:
            rayleigh_gate, rayleigh_gap, norm_error = calculate_rayleigh_reliability_gate(
                rq_energy_a,
                rq_energy_b,
                norm_train,
                virial_loss,
                rayleigh_consistency_loss,
            )
        rayleigh_advantage_loss, rayleigh_advantage_raw = calculate_rayleigh_advantage_loss(
            rq_energy, e_bar
        )
        orthogonality_loss = torch.tensor(0.0, device=device)
        rayleigh_weight = weights[6].to(torch.float32) * rayleigh_gate.to(torch.float32)
        rayleigh_objective = rq_energy if args.energy_norm_only else rayleigh_advantage_loss
        loss = (
            weights[0].to(torch.float32) * residual_loss
            + weights[1].to(torch.float32) * magnitude_loss
            + weights[2].to(torch.float32) * orthogonality_loss
            + weights[3].to(torch.float32) * virial_loss
            + weights[4].to(torch.float32) * variance_loss
            + weights[5].to(torch.float32) * rayleigh_consistency_loss
            + rayleigh_weight * rayleigh_objective
        )
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optim.step()
        optim.zero_grad()
        scheduler.step()
        end = time.time()

        norm_audit = float("nan")
        if i % norm_audit_every == 0 or i == steps - 1:
            norm_audit = float(
                estimate_norm_qmc(
                    model=model,
                    num_points=norm_audit_points,
                    box_length=L,
                    sobol_engine=sobol_engine_norm_audit,
                    nuclei_distance=R,
                    axis=axis,
                    device=device,
                ).detach().cpu()
            )

        print(
            f"Step: {i+1} | E_Rayleigh_elec: {rq_energy.item()} | E_Rayleigh_elec_A: {rq_energy_a.item()} | E_Rayleigh_elec_B: {rq_energy_b.item()} "
            f"| E_bar: {e_bar.item()} | E_Rayleigh_total: {rq_energy_total.item()} | LR: {scheduler.get_last_lr()[0]} "
            f"| Total Loss: {loss.item()} | Time: {1000*(end-start)}ms "
            f"| Residual Loss: {residual_loss.item()} | Residual Uniform: {residual_loss_uniform.item()} "
            f"| Residual Near: {residual_loss_near.item()} | Residual Norm: {residual_loss_norm.item()} "
            f"| Residual Energy: {residual_loss_energy.item()} | Lambda QMC: {lambda_qmc} "
            f"| Magnitude Loss: {magnitude_loss.item()} "
            f"| Orthogonality Loss: {orthogonality_loss.item()} | Virial Loss: {virial_loss.item()} "
            f"| Variance Loss: {variance_loss.item()} | Energy MAE: {energy_mae.item()} "
            f"| Rayleigh Consistency Loss: {rayleigh_consistency_loss.item()} "
            f"| Rayleigh Gap(A-B): {rayleigh_gap.item()} | Norm Error: {norm_error.item()} "
            f"| Rayleigh Gate: {rayleigh_gate.item()} | Rayleigh Weight Eff: {rayleigh_weight.item()} "
            f"| Rayleigh Advantage Raw: {rayleigh_advantage_raw.item()} "
            f"| Rayleigh Advantage Clipped: {rayleigh_advantage_loss.item()} "
            f"| Rayleigh Objective: {rayleigh_objective.item()} "
            f"| Norm Train(QMC): {norm_train.item()} | Norm Audit(QMC): {norm_audit} "
            f"| Jastrow b_eff: {F.softplus(model.b.detach()).item()}"
        )

    torch.save(model.state_dict(), checkpoint_path)
    print(f"Saved checkpoint to {checkpoint_path}")

    if args.skip_eval:
        print("Skipping post-training bonding_analysis-style evaluation because --skip-eval was set.")
        return

    eval_device = device if args.eval_device == "same" else args.eval_device
    energy, error, _ = evaluate_h2_checkpoint(
        checkpoint_path=checkpoint_path,
        nuclei_distance=R,
        axis=axis,
        device=eval_device,
        box_length=args.eval_box_length,
        batch_size=args.eval_batch_size,
        n_repeats=args.eval_repeats,
    )
    print(
        f"Reference BO total energy at R={R:g}: {reference_info['energy_hartree']:.15f} Hartree.\nPredicted energy: {energy}\nError: {error}"
    )


if __name__ == "__main__":
    main()
