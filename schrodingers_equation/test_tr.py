import torch

import torch.nn.functional as F

def calculate_energy_trust_region_loss(
    rayleigh_energy,
    lower_bound=-2.0,
    upper_bound=-1.6,
    transition=0.03,
    low_penalty_weight=20.0,
    high_penalty_weight=2.0,
):
    # Reward lower energy only inside the trust region, and penalize leaving it.
    trust_gate = torch.sigmoid((rayleigh_energy - lower_bound) / transition) * torch.sigmoid(
        (upper_bound - rayleigh_energy) / transition
    )
    region_reward = trust_gate * rayleigh_energy

    low_violation = F.relu(lower_bound - rayleigh_energy)
    high_violation = F.relu(rayleigh_energy - upper_bound)
    boundary_penalty = low_penalty_weight * low_violation.pow(2) + high_penalty_weight * high_violation.pow(2)

    return region_reward + boundary_penalty, trust_gate, low_violation, high_violation

print(calculate_energy_trust_region_loss(torch.tensor(1.0))[0])
print(calculate_energy_trust_region_loss(torch.tensor(-1.5))[0])
print(calculate_energy_trust_region_loss(torch.tensor(-1.88876071429))[0])
print(calculate_energy_trust_region_loss(torch.tensor(-4.0))[0])