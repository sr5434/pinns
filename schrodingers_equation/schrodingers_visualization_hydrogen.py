import argparse
import pathlib
import sys
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from schrodingers_equation_hydrogen import SchrodingerEquationHydrogen


def get_device(prefer_mps=True, prefer_cuda=True):
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_model_density(model, device, r_max, grid_points, z0=0.0, n_value=1, l_value=0, m_value=0):
    x = torch.linspace(-r_max, r_max, grid_points, device=device)
    y = torch.linspace(-r_max, r_max, grid_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z0)

    radial = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2).clamp_min(1e-6).unsqueeze(-1)
    cos_theta = (zz / radial.squeeze(-1)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta).unsqueeze(-1)
    phi = torch.atan2(yy, xx).unsqueeze(-1)

    n = torch.full_like(radial, float(n_value))
    l = torch.full_like(n, float(l_value))
    m = torch.full_like(n, float(m_value))
    with torch.no_grad():
        _, psi = model(radial, theta, phi, n, l, m)
        density = psi.pow(2).sum(dim=-1).view(grid_points, grid_points).cpu()
    return density


def estimate_model_normalization(model, device, n_value, l_value, m_value, r_max=10.0, samples=200_000):
    """Monte Carlo estimate of ∫|psi|^2 dV inside a ball of radius r_max (uniform-in-volume)."""
    epsilon = 1e-6
    # Uniform-in-volume: r = r_max * U^(1/3)
    r = (r_max - epsilon) * torch.rand(samples, 1, device=device).pow(1.0 / 3.0) + epsilon
    mu = 2.0 * torch.rand(samples, 1, device=device) - 1.0
    theta = torch.acos(mu)
    phi = torch.rand(samples, 1, device=device) * (2.0 * torch.pi)

    n = torch.full_like(r, float(n_value))
    l = torch.full_like(r, float(l_value))
    m = torch.full_like(r, float(m_value))

    with torch.no_grad():
        _, psi = model(r, theta, phi, n, l, m)
        prob_density = psi.pow(2).sum(dim=-1)

    volume = 4.0 * torch.pi * (r_max ** 3) / 3.0
    integral = prob_density.mean().item() * volume
    return integral


def compute_analytic_density(
    r_max,
    grid_points,
    z0=0.0,
    device="cpu",
    n_value=1,
    l_value=0,
    m_value=0,
    harmonic_convention="complex",
):
    x = torch.linspace(-r_max, r_max, grid_points, device=device)
    y = torch.linspace(-r_max, r_max, grid_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z0)
    radial = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    radial_safe = radial.clamp_min(1e-6)
    cos_theta = (zz / radial_safe).clamp(-1.0, 1.0)
    sin2_theta = (1.0 - cos_theta ** 2).clamp_min(0.0)
    phi = torch.atan2(yy, xx)

    if l_value < 0 or abs(m_value) > l_value:
        raise ValueError(f"Invalid quantum numbers: l={l_value}, m={m_value}")

    if n_value == 1 and l_value == 0 and m_value == 0:
        # |psi_100|^2 = 1/pi * exp(-2r)
        density = (1.0 / torch.pi) * torch.exp(-2.0 * radial)
    elif n_value == 2 and l_value == 0 and m_value == 0:
        # |psi_200|^2 = 1/(32*pi) * (2 - r)^2 * exp(-r)
        density = (1.0 / (32.0 * torch.pi)) * (2.0 - radial) ** 2 * torch.exp(-radial)
    elif n_value == 2 and l_value == 1 and abs(m_value) <= 1:
        # |psi_21m|^2 = (1/(32*pi)) r^2 exp(-r) cos^2(theta) for m=0
        # For |m|=1 there are two common conventions:
        # - complex Y_{1,±1} => |psi|^2 = (1/(64*pi)) r^2 exp(-r) sin^2(theta) (no phi dependence)
        # - real (p_x / p_y) combinations => |psi|^2 = (1/(32*pi)) r^2 exp(-r) sin^2(theta) cos^2(phi/sin^2(phi))
        radial_term = radial ** 2 * torch.exp(-radial)
        if m_value == 0:
            density = (1.0 / (32.0 * torch.pi)) * radial_term * cos_theta ** 2
        else:
            convention = (harmonic_convention or "complex").lower()
            if convention not in ("complex", "real"):
                raise ValueError(
                    f"Unknown harmonic_convention={harmonic_convention!r}; expected 'complex' or 'real'"
                )
            if convention == "complex":
                density = (1.0 / (64.0 * torch.pi)) * radial_term * sin2_theta
            else:
                # Match the model's real-valued basis:
                # m=+1 -> sin(theta) cos(phi) * sqrt(3/(4*pi))
                # m=-1 -> sin(theta) sin(phi) * sqrt(3/(4*pi))
                if m_value == 1:
                    density = (1.0 / (32.0 * torch.pi)) * radial_term * sin2_theta * torch.cos(phi) ** 2
                elif m_value == -1:
                    density = (1.0 / (32.0 * torch.pi)) * radial_term * sin2_theta * torch.sin(phi) ** 2
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=1")
    elif n_value == 3 and l_value == 0 and m_value == 0:
        # |psi_300|^2 = 1/(19683*pi) * (27 - 18r + 2r^2)^2 * exp(-2r/3)
        poly = 27.0 - 18.0 * radial + 2.0 * radial ** 2
        density = (1.0 / (19683.0 * torch.pi)) * poly ** 2 * torch.exp(-2.0 * radial / 3.0)
    elif n_value == 3 and l_value == 1 and abs(m_value) <= 1:
        # |R_31|^2 = (8/19683) r^2 (6 - r)^2 exp(-2r/3)
        radial_term = radial ** 2 * (6.0 - radial) ** 2 * torch.exp(-2.0 * radial / 3.0)
        if m_value == 0:
            density = (2.0 / (6561.0 * torch.pi)) * radial_term * cos_theta ** 2
        else:
            convention = (harmonic_convention or "complex").lower()
            if convention not in ("complex", "real"):
                raise ValueError(
                    f"Unknown harmonic_convention={harmonic_convention!r}; expected 'complex' or 'real'"
                )
            if convention == "complex":
                density = (3.0 / (19683.0 * torch.pi)) * radial_term * sin2_theta
            else:
                if m_value == 1:
                    density = (2.0 / (6561.0 * torch.pi)) * radial_term * sin2_theta * torch.cos(phi) ** 2
                elif m_value == -1:
                    density = (2.0 / (6561.0 * torch.pi)) * radial_term * sin2_theta * torch.sin(phi) ** 2
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=1")
    elif n_value == 3 and l_value == 2 and abs(m_value) <= 2:
        # |R_32|^2 = (8/98415) r^4 exp(-2r/3)
        radial_term = radial ** 4 * torch.exp(-2.0 * radial / 3.0)
        if m_value == 0:
            density = (1.0 / (39366.0 * torch.pi)) * radial_term * (3.0 * cos_theta ** 2 - 1.0) ** 2
        else:
            convention = (harmonic_convention or "complex").lower()
            if convention not in ("complex", "real"):
                raise ValueError(
                    f"Unknown harmonic_convention={harmonic_convention!r}; expected 'complex' or 'real'"
                )
            if convention == "complex":
                if abs(m_value) == 1:
                    density = (1.0 / (6561.0 * torch.pi)) * radial_term * sin2_theta * cos_theta ** 2
                elif abs(m_value) == 2:
                    density = (1.0 / (26244.0 * torch.pi)) * radial_term * sin2_theta ** 2
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=2")
            else:
                if m_value == 1:
                    density = (
                        (2.0 / (6561.0 * torch.pi))
                        * radial_term
                        * sin2_theta
                        * cos_theta ** 2
                        * torch.cos(phi) ** 2
                    )
                elif m_value == -1:
                    density = (
                        (2.0 / (6561.0 * torch.pi))
                        * radial_term
                        * sin2_theta
                        * cos_theta ** 2
                        * torch.sin(phi) ** 2
                    )
                elif m_value == 2:
                    density = (
                        (1.0 / (13122.0 * torch.pi))
                        * radial_term
                        * sin2_theta ** 2
                        * torch.cos(2.0 * phi) ** 2
                    )
                elif m_value == -2:
                    density = (
                        (1.0 / (13122.0 * torch.pi))
                        * radial_term
                        * sin2_theta ** 2
                        * torch.sin(2.0 * phi) ** 2
                    )
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=2")
    else:
        raise ValueError(f"Analytic density not implemented for n={n_value}, l={l_value}, m={m_value}")
    return density.cpu()


def compute_analytic_density_3d(
    coords,
    n_value=1,
    l_value=0,
    m_value=0,
    harmonic_convention="complex",
):
    coords_t = torch.as_tensor(coords, dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(coords_t, coords_t, coords_t, indexing="ij")
    radial = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2)
    radial_safe = radial.clamp_min(1e-6)
    cos_theta = (zz / radial_safe).clamp(-1.0, 1.0)
    sin2_theta = (1.0 - cos_theta ** 2).clamp_min(0.0)
    phi = torch.atan2(yy, xx)

    if l_value < 0 or abs(m_value) > l_value:
        raise ValueError(f"Invalid quantum numbers: l={l_value}, m={m_value}")

    if n_value == 1 and l_value == 0 and m_value == 0:
        density = (1.0 / torch.pi) * torch.exp(-2.0 * radial)
    elif n_value == 2 and l_value == 0 and m_value == 0:
        density = (1.0 / (32.0 * torch.pi)) * (2.0 - radial) ** 2 * torch.exp(-radial)
    elif n_value == 2 and l_value == 1 and abs(m_value) <= 1:
        radial_term = radial ** 2 * torch.exp(-radial)
        if m_value == 0:
            density = (1.0 / (32.0 * torch.pi)) * radial_term * cos_theta ** 2
        else:
            convention = (harmonic_convention or "complex").lower()
            if convention not in ("complex", "real"):
                raise ValueError(
                    f"Unknown harmonic_convention={harmonic_convention!r}; expected 'complex' or 'real'"
                )
            if convention == "complex":
                density = (1.0 / (64.0 * torch.pi)) * radial_term * sin2_theta
            else:
                if m_value == 1:
                    density = (1.0 / (32.0 * torch.pi)) * radial_term * sin2_theta * torch.cos(phi) ** 2
                elif m_value == -1:
                    density = (1.0 / (32.0 * torch.pi)) * radial_term * sin2_theta * torch.sin(phi) ** 2
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=1")
    elif n_value == 3 and l_value == 0 and m_value == 0:
        poly = 27.0 - 18.0 * radial + 2.0 * radial ** 2
        density = (1.0 / (19683.0 * torch.pi)) * poly ** 2 * torch.exp(-2.0 * radial / 3.0)
    elif n_value == 3 and l_value == 1 and abs(m_value) <= 1:
        radial_term = radial ** 2 * (6.0 - radial) ** 2 * torch.exp(-2.0 * radial / 3.0)
        if m_value == 0:
            density = (2.0 / (6561.0 * torch.pi)) * radial_term * cos_theta ** 2
        else:
            convention = (harmonic_convention or "complex").lower()
            if convention not in ("complex", "real"):
                raise ValueError(
                    f"Unknown harmonic_convention={harmonic_convention!r}; expected 'complex' or 'real'"
                )
            if convention == "complex":
                density = (3.0 / (19683.0 * torch.pi)) * radial_term * sin2_theta
            else:
                if m_value == 1:
                    density = (2.0 / (6561.0 * torch.pi)) * radial_term * sin2_theta * torch.cos(phi) ** 2
                elif m_value == -1:
                    density = (2.0 / (6561.0 * torch.pi)) * radial_term * sin2_theta * torch.sin(phi) ** 2
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=1")
    elif n_value == 3 and l_value == 2 and abs(m_value) <= 2:
        radial_term = radial ** 4 * torch.exp(-2.0 * radial / 3.0)
        if m_value == 0:
            density = (1.0 / (39366.0 * torch.pi)) * radial_term * (3.0 * cos_theta ** 2 - 1.0) ** 2
        else:
            convention = (harmonic_convention or "complex").lower()
            if convention not in ("complex", "real"):
                raise ValueError(
                    f"Unknown harmonic_convention={harmonic_convention!r}; expected 'complex' or 'real'"
                )
            if convention == "complex":
                if abs(m_value) == 1:
                    density = (1.0 / (6561.0 * torch.pi)) * radial_term * sin2_theta * cos_theta ** 2
                elif abs(m_value) == 2:
                    density = (1.0 / (26244.0 * torch.pi)) * radial_term * sin2_theta ** 2
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=2")
            else:
                if m_value == 1:
                    density = (
                        (2.0 / (6561.0 * torch.pi))
                        * radial_term
                        * sin2_theta
                        * cos_theta ** 2
                        * torch.cos(phi) ** 2
                    )
                elif m_value == -1:
                    density = (
                        (2.0 / (6561.0 * torch.pi))
                        * radial_term
                        * sin2_theta
                        * cos_theta ** 2
                        * torch.sin(phi) ** 2
                    )
                elif m_value == 2:
                    density = (
                        (1.0 / (13122.0 * torch.pi))
                        * radial_term
                        * sin2_theta ** 2
                        * torch.cos(2.0 * phi) ** 2
                    )
                elif m_value == -2:
                    density = (
                        (1.0 / (13122.0 * torch.pi))
                        * radial_term
                        * sin2_theta ** 2
                        * torch.sin(2.0 * phi) ** 2
                    )
                else:
                    raise ValueError(f"Unexpected m={m_value} for l=2")
    else:
        raise ValueError(f"Analytic density not implemented for n={n_value}, l={l_value}, m={m_value}")
    return density.numpy()


def compute_model_density_3d(
    model,
    device,
    r_max,
    grid_points,
    n_value,
    l_value,
    m_value,
    batch_size=200_000,
):
    coords = torch.linspace(-r_max, r_max, grid_points, device="cpu")
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    coords_flat = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
    densities = torch.empty(coords_flat.shape[0], device="cpu")

    with torch.no_grad():
        for start in range(0, coords_flat.shape[0], batch_size):
            batch = coords_flat[start : start + batch_size].to(device)
            r = torch.linalg.norm(batch, dim=1, keepdim=True).clamp_min(1e-6)
            cos_theta = (batch[:, 2:3] / r).clamp(-1.0, 1.0)
            theta = torch.acos(cos_theta)
            phi = torch.atan2(batch[:, 1:2], batch[:, 0:1])

            n = torch.full_like(r, float(n_value))
            l = torch.full_like(r, float(l_value))
            m = torch.full_like(r, float(m_value))

            _, psi = model(r, theta, phi, n, l, m)
            density = psi.pow(2).sum(dim=-1)
            densities[start : start + batch_size] = density.cpu()

    density_grid = densities.view(grid_points, grid_points, grid_points).numpy()
    return coords.numpy(), density_grid


def get_h2_nuclei_positions(axis, nuclei_distance, device="cpu", dtype=torch.float32):
    half = 0.5 * float(nuclei_distance)
    nucleus_a = torch.zeros(3, device=device, dtype=dtype)
    nucleus_b = torch.zeros(3, device=device, dtype=dtype)
    if axis == "x":
        nucleus_a[0], nucleus_b[0] = -half, half
    elif axis == "y":
        nucleus_a[1], nucleus_b[1] = -half, half
    elif axis == "z":
        nucleus_a[2], nucleus_b[2] = -half, half
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    return nucleus_a, nucleus_b


def analytic_1s_wavefunction(r):
    return torch.exp(-r) / np.sqrt(np.pi)


def analytic_h2_overlap_1s(nuclei_distance):
    r = torch.as_tensor(float(nuclei_distance))
    return torch.exp(-r) * (1.0 + r + (r ** 2) / 3.0)


def evaluate_model_1s(model, rel_points):
    r = torch.linalg.norm(rel_points, dim=-1, keepdim=True).clamp_min(1e-6)
    cos_theta = (rel_points[:, 2:3] / r).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta)
    phi = torch.atan2(rel_points[:, 1:2], rel_points[:, 0:1])

    n = torch.ones_like(r)
    l = torch.zeros_like(r)
    m = torch.zeros_like(r)

    _, psi = model(r, theta, phi, n, l, m)
    return psi


def estimate_model_h2_overlap(
    model,
    device,
    extent,
    axis,
    nuclei_distance,
    samples=200_000,
    batch_size=200_000,
):
    nucleus_a, nucleus_b = get_h2_nuclei_positions(axis, nuclei_distance, device=device)
    volume = (2.0 * extent) ** 3
    overlap_accum = 0.0
    samples = int(samples)

    with torch.no_grad():
        for start in range(0, samples, batch_size):
            count = min(batch_size, samples - start)
            points = (torch.rand(count, 3, device=device) - 0.5) * (2.0 * extent)
            rel_a = points - nucleus_a
            rel_b = points - nucleus_b
            psi_a = evaluate_model_1s(model, rel_a)
            psi_b = evaluate_model_1s(model, rel_b)
            overlap_accum += (psi_a * psi_b).sum(dim=-1).sum().item()

    return (overlap_accum / samples) * volume


def compute_h2_lcao_model_density_slice(
    model,
    device,
    extent,
    grid_points,
    z0,
    axis,
    nuclei_distance,
    state,
    overlap=None,
    overlap_samples=200_000,
    overlap_batch=200_000,
):
    if overlap is None:
        overlap = estimate_model_h2_overlap(
            model,
            device,
            extent=extent,
            axis=axis,
            nuclei_distance=nuclei_distance,
            samples=overlap_samples,
            batch_size=overlap_batch,
        )

    x = torch.linspace(-extent, extent, grid_points, device=device)
    y = torch.linspace(-extent, extent, grid_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, float(z0))
    coords = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)

    nucleus_a, nucleus_b = get_h2_nuclei_positions(axis, nuclei_distance, device=device)
    rel_a = coords - nucleus_a
    rel_b = coords - nucleus_b

    with torch.no_grad():
        psi_a = evaluate_model_1s(model, rel_a)
        psi_b = evaluate_model_1s(model, rel_b)
        if state == "bonding":
            norm = np.sqrt(2.0 * max(1e-8, 1.0 + overlap))
            psi = (psi_a + psi_b) / norm
        elif state == "antibonding":
            norm = np.sqrt(2.0 * max(1e-8, 1.0 - overlap))
            psi = (psi_a - psi_b) / norm
        else:
            raise ValueError("state must be 'bonding' or 'antibonding'")
        density = psi.pow(2).sum(dim=-1)

    return density.view(grid_points, grid_points).cpu(), overlap


def compute_h2_lcao_analytic_density_slice(
    extent,
    grid_points,
    z0,
    axis,
    nuclei_distance,
    state,
    overlap=None,
    device="cpu",
):
    x = torch.linspace(-extent, extent, grid_points, device=device)
    y = torch.linspace(-extent, extent, grid_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, float(z0))

    nucleus_a, nucleus_b = get_h2_nuclei_positions(axis, nuclei_distance, device=device)
    rel_a = torch.stack([xx - nucleus_a[0], yy - nucleus_a[1], zz - nucleus_a[2]], dim=-1)
    rel_b = torch.stack([xx - nucleus_b[0], yy - nucleus_b[1], zz - nucleus_b[2]], dim=-1)

    r_a = torch.linalg.norm(rel_a, dim=-1).clamp_min(1e-6)
    r_b = torch.linalg.norm(rel_b, dim=-1).clamp_min(1e-6)
    psi_a = analytic_1s_wavefunction(r_a)
    psi_b = analytic_1s_wavefunction(r_b)

    if overlap is None:
        overlap = analytic_h2_overlap_1s(nuclei_distance)
    overlap = torch.as_tensor(overlap, device=psi_a.device, dtype=psi_a.dtype)

    if state == "bonding":
        norm = torch.sqrt(2.0 * (1.0 + overlap))
        psi = (psi_a + psi_b) / norm
    elif state == "antibonding":
        norm = torch.sqrt(2.0 * (1.0 - overlap))
        psi = (psi_a - psi_b) / norm
    else:
        raise ValueError("state must be 'bonding' or 'antibonding'")

    density = psi.pow(2)
    return density.cpu(), float(overlap)


def compute_h2_lcao_model_density_3d(
    model,
    device,
    extent,
    grid_points,
    axis,
    nuclei_distance,
    state,
    batch_size=200_000,
    overlap=None,
    overlap_samples=200_000,
    overlap_batch=200_000,
):
    if overlap is None:
        overlap = estimate_model_h2_overlap(
            model,
            device,
            extent=extent,
            axis=axis,
            nuclei_distance=nuclei_distance,
            samples=overlap_samples,
            batch_size=overlap_batch,
        )

    coords = torch.linspace(-extent, extent, grid_points, device="cpu")
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    coords_flat = torch.stack([xx, yy, zz], dim=-1).view(-1, 3)
    densities = torch.empty(coords_flat.shape[0], device="cpu")

    nucleus_a, nucleus_b = get_h2_nuclei_positions(axis, nuclei_distance, device=device)

    with torch.no_grad():
        for start in range(0, coords_flat.shape[0], batch_size):
            batch = coords_flat[start : start + batch_size].to(device)
            rel_a = batch - nucleus_a
            rel_b = batch - nucleus_b
            psi_a = evaluate_model_1s(model, rel_a)
            psi_b = evaluate_model_1s(model, rel_b)

            if state == "bonding":
                norm = np.sqrt(2.0 * max(1e-8, 1.0 + overlap))
                psi = (psi_a + psi_b) / norm
            elif state == "antibonding":
                norm = np.sqrt(2.0 * max(1e-8, 1.0 - overlap))
                psi = (psi_a - psi_b) / norm
            else:
                raise ValueError("state must be 'bonding' or 'antibonding'")

            density = psi.pow(2).sum(dim=-1)
            densities[start : start + batch_size] = density.cpu()

    density_grid = densities.view(grid_points, grid_points, grid_points).numpy()
    return coords.numpy(), density_grid, overlap


def compute_h2_lcao_analytic_density_3d(
    coords,
    axis,
    nuclei_distance,
    state,
    overlap=None,
):
    coords_t = torch.as_tensor(coords, dtype=torch.float32)
    xx, yy, zz = torch.meshgrid(coords_t, coords_t, coords_t, indexing="ij")
    nucleus_a, nucleus_b = get_h2_nuclei_positions(axis, nuclei_distance, device=coords_t.device)
    rel_a = torch.stack([xx - nucleus_a[0], yy - nucleus_a[1], zz - nucleus_a[2]], dim=-1)
    rel_b = torch.stack([xx - nucleus_b[0], yy - nucleus_b[1], zz - nucleus_b[2]], dim=-1)

    r_a = torch.linalg.norm(rel_a, dim=-1).clamp_min(1e-6)
    r_b = torch.linalg.norm(rel_b, dim=-1).clamp_min(1e-6)
    psi_a = analytic_1s_wavefunction(r_a)
    psi_b = analytic_1s_wavefunction(r_b)

    if overlap is None:
        overlap = analytic_h2_overlap_1s(nuclei_distance)
    overlap = torch.as_tensor(overlap, device=psi_a.device, dtype=psi_a.dtype)

    if state == "bonding":
        norm = torch.sqrt(2.0 * (1.0 + overlap))
        psi = (psi_a + psi_b) / norm
    elif state == "antibonding":
        norm = torch.sqrt(2.0 * (1.0 - overlap))
        psi = (psi_a - psi_b) / norm
    else:
        raise ValueError("state must be 'bonding' or 'antibonding'")

    density = psi.pow(2)
    return density.numpy(), float(overlap)


def sample_orbital_points(
    coords,
    density_grid,
    r_max,
    iso_quantile=0.996,
    iso_value=None,
    max_points=120_000,
    seed=7,
):
    coord2 = coords.astype(np.float32) ** 2
    r2 = (
        coord2[:, None, None]
        + coord2[None, :, None]
        + coord2[None, None, :]
    )
    sphere_mask = r2 <= (r_max ** 2 + 1e-6)
    valid_density = density_grid[sphere_mask]
    if valid_density.size == 0:
        raise ValueError("No density samples inside r_max; increase grid or r_max.")

    if iso_value is None:
        threshold = float(np.quantile(valid_density, iso_quantile))
    else:
        threshold = float(iso_value)

    mask = (density_grid >= threshold) & sphere_mask
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError(
            "No points above the isosurface threshold; lower --iso_quantile or --iso_value."
        )

    if max_points and indices.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        choice = rng.choice(indices.shape[0], size=max_points, replace=False)
        indices = indices[choice]

    x = coords[indices[:, 0]]
    y = coords[indices[:, 1]]
    z = coords[indices[:, 2]]
    values = density_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    return x, y, z, values, threshold


def resolve_output_path(base_output, n_value, l_value, m_value):
    base = pathlib.Path(base_output)
    base_str = str(base)
    if any(token in base_str for token in ("{n}", "{l}", "{m}", "{orbital}")):
        orbital = format_orbital_label(n_value, l_value, m_value).replace(" ", "")
        return pathlib.Path(base_str.format(n=n_value, l=l_value, m=m_value, orbital=orbital))
    if base.suffix:
        return base.with_name(f"{base.stem}_n{n_value}_l{l_value}_m{m_value}{base.suffix}")
    return base / f"n{n_value}_l{l_value}_m{m_value}.png"


def format_orbital_label(n_value, l_value, m_value):
    letters = {0: "s", 1: "p", 2: "d", 3: "f"}
    l_label = letters.get(l_value, f"l{l_value}")
    return f"{n_value}{l_label} (m={m_value})"


def format_h2_label(state, axis, nuclei_distance):
    return f"H2+ {state} (R={nuclei_distance:g} a.u., axis={axis})"


def resolve_h2_output_path_with_suffix(base_output, state, axis, nuclei_distance, default_suffix):
    base = pathlib.Path(base_output)
    base_str = str(base)
    dist_str = f"{nuclei_distance:g}"
    dist_slug = dist_str.replace(".", "p")
    orbital = f"h2_{state}_R{dist_slug}_{axis}"
    if any(token in base_str for token in ("{state}", "{axis}", "{R}", "{orbital}")):
        candidate = pathlib.Path(
            base_str.format(state=state, axis=axis, R=dist_str, orbital=orbital)
        )
        if candidate.suffix:
            return candidate
        return candidate.with_suffix(default_suffix)
    if base.suffix:
        return base.with_name(f"{base.stem}_{orbital}{base.suffix}")
    return (base / orbital).with_suffix(default_suffix)


def resolve_output_path_with_suffix(base_output, n_value, l_value, m_value, default_suffix):
    base = pathlib.Path(base_output)
    base_str = str(base)
    if any(token in base_str for token in ("{n}", "{l}", "{m}", "{orbital}")):
        orbital = format_orbital_label(n_value, l_value, m_value).replace(" ", "")
        candidate = pathlib.Path(base_str.format(n=n_value, l=l_value, m=m_value, orbital=orbital))
        if candidate.suffix:
            return candidate
        return candidate.with_suffix(default_suffix)
    if base.suffix:
        return base.with_name(f"{base.stem}_n{n_value}_l{l_value}_m{m_value}{base.suffix}")
    return (base / f"n{n_value}_l{l_value}_m{m_value}").with_suffix(default_suffix)


def render_orbital_video(
    model,
    device,
    output_path,
    r_max,
    grid_points,
    n_value,
    l_value,
    m_value,
    frames=180,
    fps=30,
    dpi=160,
    batch_size=200_000,
    iso_quantile=0.996,
    iso_value=None,
    max_points=120_000,
    seed=7,
    point_size=0.6,
    alpha=0.85,
    cmap="magma",
    elev=20.0,
    elev_amplitude=10.0,
    spin_degrees=360.0,
    show_axes=False,
):
    coords, density_grid = compute_model_density_3d(
        model,
        device,
        r_max=r_max,
        grid_points=grid_points,
        n_value=n_value,
        l_value=l_value,
        m_value=m_value,
        batch_size=batch_size,
    )
    x, y, z, values, threshold = sample_orbital_points(
        coords,
        density_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )

    label = format_orbital_label(n_value, l_value, m_value)
    print(f"Rendering {label} with {x.shape[0]} points (threshold={threshold:.3e})...")

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    sc = ax.scatter(
        x,
        y,
        z,
        c=values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
    )

    ax.set_xlim(-r_max, r_max)
    ax.set_ylim(-r_max, r_max)
    ax.set_zlim(-r_max, r_max)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"Hydrogen {label} |psi|^2", color="white")

    if not show_axes:
        ax.set_axis_off()
    else:
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

    def update(frame):
        azim = (spin_degrees * frame / frames) % 360.0
        elev_current = elev + elev_amplitude * np.sin(2.0 * np.pi * frame / frames)
        ax.view_init(elev=elev_current, azim=azim)
        return (sc,)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    try:
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "Failed to write video. Ensure ffmpeg is installed or try a different output path."
        ) from exc
    plt.close(fig)
    return output_path


def render_orbital_comparison_video(
    model,
    device,
    output_path,
    r_max,
    grid_points,
    n_value,
    l_value,
    m_value,
    frames=180,
    fps=30,
    dpi=160,
    batch_size=200_000,
    iso_quantile=0.996,
    iso_value=None,
    max_points=120_000,
    seed=7,
    point_size=0.6,
    alpha=0.85,
    cmap="magma",
    error_cmap="viridis",
    elev=20.0,
    elev_amplitude=10.0,
    spin_degrees=360.0,
    show_axes=False,
    harmonic_convention=None,
):
    coords, model_density_grid = compute_model_density_3d(
        model,
        device,
        r_max=r_max,
        grid_points=grid_points,
        n_value=n_value,
        l_value=l_value,
        m_value=m_value,
        batch_size=batch_size,
    )
    if harmonic_convention is None:
        harmonic_convention = getattr(model, "harmonics", "complex")
    analytic_density_grid = compute_analytic_density_3d(
        coords,
        n_value=n_value,
        l_value=l_value,
        m_value=m_value,
        harmonic_convention=harmonic_convention,
    )
    error_grid = np.abs(model_density_grid - analytic_density_grid)
    mae = float(error_grid.mean())

    model_x, model_y, model_z, model_values, model_threshold = sample_orbital_points(
        coords,
        model_density_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )
    analytic_x, analytic_y, analytic_z, analytic_values, analytic_threshold = sample_orbital_points(
        coords,
        analytic_density_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )
    error_x, error_y, error_z, error_values, error_threshold = sample_orbital_points(
        coords,
        error_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=None,
        max_points=max_points,
        seed=seed,
    )

    label = format_orbital_label(n_value, l_value, m_value)
    density_vmax = float(max(model_density_grid.max(), analytic_density_grid.max()))
    error_vmax = float(error_grid.max())
    if error_vmax <= 0.0:
        error_vmax = 1e-12
    print(
        "Rendering comparison video for "
        f"{label} | model pts={model_x.shape[0]} (thr={model_threshold:.3e}), "
        f"analytic pts={analytic_x.shape[0]} (thr={analytic_threshold:.3e}), "
        f"error pts={error_x.shape[0]} (thr={error_threshold:.3e})..."
    )

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor("black")
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    titles = [
        f"Model {label} |psi|^2",
        "Analytic |psi|^2",
        "Absolute error",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor("black")
        ax.set_xlim(-r_max, r_max)
        ax.set_ylim(-r_max, r_max)
        ax.set_zlim(-r_max, r_max)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(title, color="white")
        if not show_axes:
            ax.set_axis_off()
        else:
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")

    sc_model = axes[0].scatter(
        model_x,
        model_y,
        model_z,
        c=model_values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=density_vmax,
    )
    sc_analytic = axes[1].scatter(
        analytic_x,
        analytic_y,
        analytic_z,
        c=analytic_values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=density_vmax,
    )
    sc_error = axes[2].scatter(
        error_x,
        error_y,
        error_z,
        c=error_values,
        s=point_size,
        cmap=error_cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=error_vmax,
    )

    def update(frame):
        azim = (spin_degrees * frame / frames) % 360.0
        elev_current = elev + elev_amplitude * np.sin(2.0 * np.pi * frame / frames)
        for ax in axes:
            ax.view_init(elev=elev_current, azim=azim)
        return (sc_model, sc_analytic, sc_error)

    fig.text(
        0.5,
        0.02,
        f"MAE: {mae:.6e}",
        ha="center",
        va="bottom",
        color="white",
        fontsize=10,
    )

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    try:
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "Failed to write video. Ensure ffmpeg is installed or try a different output path."
        ) from exc
    plt.close(fig)
    return output_path


def render_h2_orbital_video(
    model,
    device,
    output_path,
    extent,
    grid_points,
    axis,
    nuclei_distance,
    state,
    frames=180,
    fps=30,
    dpi=160,
    batch_size=200_000,
    iso_quantile=0.996,
    iso_value=None,
    max_points=120_000,
    seed=7,
    point_size=0.6,
    alpha=0.85,
    cmap="magma",
    elev=20.0,
    elev_amplitude=10.0,
    spin_degrees=360.0,
    show_axes=False,
    overlap_samples=200_000,
    overlap_batch=200_000,
):
    coords, density_grid, overlap = compute_h2_lcao_model_density_3d(
        model,
        device,
        extent=extent,
        grid_points=grid_points,
        axis=axis,
        nuclei_distance=nuclei_distance,
        state=state,
        batch_size=batch_size,
        overlap_samples=overlap_samples,
        overlap_batch=overlap_batch,
    )
    x, y, z, values, threshold = sample_orbital_points(
        coords,
        density_grid,
        r_max=extent,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )

    label = format_h2_label(state, axis, nuclei_distance)
    print(f"Rendering {label} with {x.shape[0]} points (S={overlap:.4f}, thr={threshold:.3e})...")

    fig = plt.figure(figsize=(6.5, 6.5))
    ax = fig.add_subplot(111, projection="3d")
    fig.patch.set_facecolor("black")
    ax.set_facecolor("black")

    sc = ax.scatter(
        x,
        y,
        z,
        c=values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
    )

    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_box_aspect((1, 1, 1))
    ax.set_title(f"{label} |psi|^2", color="white")

    if not show_axes:
        ax.set_axis_off()
    else:
        for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis_obj.set_tick_params(colors="white")
        ax.xaxis.label.set_color("white")
        ax.yaxis.label.set_color("white")
        ax.zaxis.label.set_color("white")

    def update(frame):
        azim = (spin_degrees * frame / frames) % 360.0
        elev_current = elev + elev_amplitude * np.sin(2.0 * np.pi * frame / frames)
        ax.view_init(elev=elev_current, azim=azim)
        return (sc,)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    try:
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "Failed to write video. Ensure ffmpeg is installed or try a different output path."
        ) from exc
    plt.close(fig)
    return output_path


def render_h2_orbital_comparison_video(
    model,
    device,
    output_path,
    extent,
    grid_points,
    axis,
    nuclei_distance,
    state,
    frames=180,
    fps=30,
    dpi=160,
    batch_size=200_000,
    iso_quantile=0.996,
    iso_value=None,
    max_points=120_000,
    seed=7,
    point_size=0.6,
    alpha=0.85,
    cmap="magma",
    error_cmap="viridis",
    elev=20.0,
    elev_amplitude=10.0,
    spin_degrees=360.0,
    show_axes=False,
    overlap_samples=200_000,
    overlap_batch=200_000,
):
    coords, model_density_grid, overlap = compute_h2_lcao_model_density_3d(
        model,
        device,
        extent=extent,
        grid_points=grid_points,
        axis=axis,
        nuclei_distance=nuclei_distance,
        state=state,
        batch_size=batch_size,
        overlap_samples=overlap_samples,
        overlap_batch=overlap_batch,
    )
    analytic_density_grid, analytic_overlap = compute_h2_lcao_analytic_density_3d(
        coords,
        axis=axis,
        nuclei_distance=nuclei_distance,
        state=state,
        overlap=None,
    )
    error_grid = np.abs(model_density_grid - analytic_density_grid)
    mae = float(error_grid.mean())

    model_x, model_y, model_z, model_values, model_threshold = sample_orbital_points(
        coords,
        model_density_grid,
        r_max=extent,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )
    analytic_x, analytic_y, analytic_z, analytic_values, analytic_threshold = sample_orbital_points(
        coords,
        analytic_density_grid,
        r_max=extent,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )
    error_x, error_y, error_z, error_values, error_threshold = sample_orbital_points(
        coords,
        error_grid,
        r_max=extent,
        iso_quantile=iso_quantile,
        iso_value=None,
        max_points=max_points,
        seed=seed,
    )

    label = format_h2_label(state, axis, nuclei_distance)
    density_vmax = float(max(model_density_grid.max(), analytic_density_grid.max()))
    error_vmax = float(error_grid.max())
    if error_vmax <= 0.0:
        error_vmax = 1e-12
    print(
        "Rendering H2+ comparison video for "
        f"{label} | model pts={model_x.shape[0]} (S={overlap:.4f}, thr={model_threshold:.3e}), "
        f"analytic pts={analytic_x.shape[0]} (S={analytic_overlap:.4f}, thr={analytic_threshold:.3e}), "
        f"error pts={error_x.shape[0]} (thr={error_threshold:.3e})..."
    )

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor("black")
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    titles = [
        f"Model {label} |psi|^2",
        "Analytic LCAO |psi|^2",
        "Absolute error",
    ]

    for ax, title in zip(axes, titles):
        ax.set_facecolor("black")
        ax.set_xlim(-extent, extent)
        ax.set_ylim(-extent, extent)
        ax.set_zlim(-extent, extent)
        ax.set_box_aspect((1, 1, 1))
        ax.set_title(title, color="white")
        if not show_axes:
            ax.set_axis_off()
        else:
            for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis_obj.set_tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")

    sc_model = axes[0].scatter(
        model_x,
        model_y,
        model_z,
        c=model_values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=density_vmax,
    )
    sc_analytic = axes[1].scatter(
        analytic_x,
        analytic_y,
        analytic_z,
        c=analytic_values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=density_vmax,
    )
    sc_error = axes[2].scatter(
        error_x,
        error_y,
        error_z,
        c=error_values,
        s=point_size,
        cmap=error_cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=error_vmax,
    )

    def update(frame):
        azim = (spin_degrees * frame / frames) % 360.0
        elev_current = elev + elev_amplitude * np.sin(2.0 * np.pi * frame / frames)
        for ax in axes:
            ax.view_init(elev=elev_current, azim=azim)
        return (sc_model, sc_analytic, sc_error)

    fig.text(
        0.5,
        0.02,
        f"MAE: {mae:.6e} | S(model)={overlap:.4f} | S(analytic)={analytic_overlap:.4f}",
        ha="center",
        va="bottom",
        color="white",
        fontsize=9,
    )

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    try:
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError(
            "Failed to write video. Ensure ffmpeg is installed or try a different output path."
        ) from exc
    plt.close(fig)
    return output_path


def render_comparison(
    model_path="schrodingers_equation_hydrogen.pt",
    output_path="hydrogen_{orbital}_compare.png",
    r_max=6.0,
    grid_points=220,
    z0=0.0,
    n_value=1,
    l_value=0,
    m_value=0,
    harmonic_convention=None,
    model=None,
    device=None,
    model_max_energy=None,
    renormalize_model=False,
    norm_rmax=10.0,
    norm_samples=200_000,
    print_norm=False,
):
    max_energy = int(model_max_energy) if model_max_energy is not None else n_value
    max_sublevels = min(max_energy, 4)-1
    max_orbital = max_energy - 1
    if device is None:
        device = get_device()
    if model is None:
        model = SchrodingerEquationHydrogen(L=1.0, max_n=max_energy, max_l=max_sublevels, max_m=max_orbital).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    # If the model advertises a harmonic basis, use it as the default for analytics.
    if harmonic_convention is None:
        harmonic_convention = getattr(model, "harmonics", "complex")

    norm_estimate = None
    if renormalize_model or print_norm:
        norm_estimate = estimate_model_normalization(
            model,
            device,
            n_value=n_value,
            l_value=l_value,
            m_value=m_value,
            r_max=norm_rmax,
            samples=norm_samples,
        )
        if print_norm:
            label = format_orbital_label(n_value, l_value, m_value)
            print(f"Estimated normalization inside r<={norm_rmax:g} for {label}: {norm_estimate:.6f}")

    model_density = compute_model_density(model, device, r_max, grid_points, z0, n_value, l_value, m_value)
    if renormalize_model and norm_estimate and norm_estimate > 0:
        model_density = model_density / norm_estimate
    analytic_density = compute_analytic_density(
        r_max,
        grid_points,
        z0,
        device,
        n_value,
        l_value,
        m_value,
        harmonic_convention=harmonic_convention,
    )
    error = (model_density - analytic_density).abs()
    label = format_orbital_label(n_value, l_value, m_value)
    analytic_max = analytic_density.max().item()
    if analytic_max > 0:
        print(f"Percentage max error for {label}: {100.0 * error.max().item() / analytic_max:.4f}%")
    else:
        print(f"Max absolute error for {label}: {error.max().item():.6e}")
    vmax = max(model_density.max().item(), analytic_density.max().item())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Model |psi|^2", f"Analytic {label} |psi|^2", "Absolute error"]
    data = [model_density, analytic_density, error]
    cmaps = ["magma", "magma", "viridis"]
    vmins = [0.0, 0.0, 0.0]
    vmaxs = [vmax, vmax, error.max().item() if error.max().item() > 0 else 1e-9]

    for ax, img, title, cmap, vmin, vmax_local in zip(axes, data, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(
            img,
            extent=[-r_max, r_max, -r_max, r_max],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_local,
        )
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(f"Hydrogen {label} density comparison at z = {z0:.2f} a.u.")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = pathlib.Path(resolve_output_path(output_path, n_value, l_value, m_value))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def render_h2_comparison(
    model_path="schrodingers_equation_hydrogen.pt",
    output_path="h2_{orbital}_compare.png",
    extent=6.0,
    grid_points=220,
    z0=0.0,
    axis="x",
    nuclei_distance=2.0,
    state="bonding",
    model=None,
    device=None,
    model_max_energy=None,
    overlap_samples=200_000,
    overlap_batch=200_000,
):
    if device is None:
        device = get_device()
    if model is None:
        max_energy = int(model_max_energy) if model_max_energy is not None else 1
        model = load_model(model_path, device, max_energy=max_energy)

    model_density, overlap = compute_h2_lcao_model_density_slice(
        model,
        device,
        extent=extent,
        grid_points=grid_points,
        z0=z0,
        axis=axis,
        nuclei_distance=nuclei_distance,
        state=state,
        overlap_samples=overlap_samples,
        overlap_batch=overlap_batch,
    )
    analytic_density, analytic_overlap = compute_h2_lcao_analytic_density_slice(
        extent=extent,
        grid_points=grid_points,
        z0=z0,
        axis=axis,
        nuclei_distance=nuclei_distance,
        state=state,
    )
    error = (model_density - analytic_density).abs()
    label = format_h2_label(state, axis, nuclei_distance)
    analytic_max = float(analytic_density.max())
    if analytic_max > 0:
        print(f"Percentage max error for {label}: {100.0 * error.max().item() / analytic_max:.4f}%")
    else:
        print(f"Max absolute error for {label}: {error.max().item():.6e}")
    vmax = max(model_density.max().item(), analytic_density.max().item())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Model LCAO |psi|^2", "Analytic LCAO |psi|^2", "Absolute error"]
    data = [model_density, analytic_density, error]
    cmaps = ["magma", "magma", "viridis"]
    vmins = [0.0, 0.0, 0.0]
    vmaxs = [vmax, vmax, error.max().item() if error.max().item() > 0 else 1e-9]

    for ax, img, title, cmap, vmin, vmax_local in zip(axes, data, titles, cmaps, vmins, vmaxs):
        im = ax.imshow(
            img,
            extent=[-extent, extent, -extent, extent],
            origin="lower",
            cmap=cmap,
            vmin=vmin,
            vmax=vmax_local,
        )
        ax.set_xlabel("x (a.u.)")
        ax.set_ylabel("y (a.u.)")
        ax.set_title(title)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"{label} density comparison at z = {z0:.2f} a.u. | S(model)={overlap:.4f} | S(analytic)={analytic_overlap:.4f}"
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    output_path = resolve_h2_output_path_with_suffix(
        output_path,
        state=state,
        axis=axis,
        nuclei_distance=nuclei_distance,
        default_suffix=".png",
    )
    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def parse_int_list(value):
    if value is None:
        return []
    items = []
    for token in value.split(","):
        token = token.strip()
        if token:
            items.append(int(token))
    return items


def load_model(model_path, device, max_energy):
    max_sublevels = min(max_energy, 4) - 1
    max_orbital = max_energy - 1
    model = SchrodingerEquationHydrogen(
        L=1.0,
        max_n=max_energy,
        max_l=max_sublevels,
        max_m=max_orbital,
    ).to(device)
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    return model


def iter_state_triplets(states, ls, ms, render_all):
    for n_value in states:
        if render_all:
            l_values = list(range(n_value))
            m_values_by_l = {l_value: list(range(-l_value, l_value + 1)) for l_value in l_values}
        else:
            l_values = ls
            m_values_by_l = {l_value: ms for l_value in l_values}

        for l_value in l_values:
            if l_value < 0 or l_value >= n_value:
                raise ValueError(f"Invalid l={l_value} for n={n_value}")
            for m_value in m_values_by_l[l_value]:
                if abs(m_value) > l_value:
                    raise ValueError(f"Invalid m={m_value} for l={l_value}")
                yield n_value, l_value, m_value


def run_default_batch():
    device = get_device()
    model_path = "schrodingers_equation_hydrogen.pt"

    h2_extent = 6.0
    h2_grid = 80
    h2_axis = "x"
    h2_distance = 2.0
    h2_output = "assets/h2_{orbital}_compare.mp4"

    h2_model = load_model(model_path, device, max_energy=1)
    for state in ("antibonding", "bonding"):
        output = resolve_h2_output_path_with_suffix(
            h2_output,
            state=state,
            axis=h2_axis,
            nuclei_distance=h2_distance,
            default_suffix=".mp4",
        )
        output = render_h2_orbital_comparison_video(
            model=h2_model,
            device=device,
            output_path=output,
            extent=h2_extent,
            grid_points=h2_grid,
            axis=h2_axis,
            nuclei_distance=h2_distance,
            state=state,
            frames=180,
            fps=30,
            dpi=160,
            batch_size=200_000,
            iso_quantile=0.996,
            iso_value=None,
            max_points=120_000,
            seed=7,
            point_size=0.6,
            alpha=0.85,
            cmap="magma",
            error_cmap="viridis",
            elev=20.0,
            elev_amplitude=10.0,
            spin_degrees=360.0,
            show_axes=False,
            overlap_samples=200_000,
            overlap_batch=200_000,
        )
        print(f"Saved H2+ comparison video for {state} to {output}")

    orbital_output = "assets/hydrogen_{orbital}_compare.mp4"
    orbital_model = load_model(model_path, device, max_energy=3)
    states = [1, 2]
    for n_value, l_value, m_value in iter_state_triplets(states, [], [], render_all=True):
        output = resolve_output_path_with_suffix(orbital_output, n_value, l_value, m_value, default_suffix=".mp4")
        output = render_orbital_comparison_video(
            model=orbital_model,
            device=device,
            output_path=output,
            r_max=6.0,
            grid_points=80,
            n_value=n_value,
            l_value=l_value,
            m_value=m_value,
            frames=180,
            fps=30,
            dpi=160,
            batch_size=200_000,
            iso_quantile=0.996,
            iso_value=None,
            max_points=120_000,
            seed=7,
            point_size=0.6,
            alpha=0.85,
            cmap="magma",
            error_cmap="viridis",
            elev=20.0,
            elev_amplitude=10.0,
            spin_degrees=360.0,
            show_axes=False,
            harmonic_convention="complex",
        )
        print(f"Saved comparison video for n={n_value}, l={l_value}, m={m_value} to {output}")


def main():
    if len(sys.argv) == 1:
        run_default_batch()
        return

    parser = argparse.ArgumentParser(
        description="Hydrogen visualization: 3D orbital videos, analytic slice comparisons, and H2+ LCAO renders."
    )
    parser.add_argument(
        "--mode",
        choices=("orbitals", "compare", "h2", "h2_compare"),
        default="orbitals",
        help="Render 3D orbital videos, analytic slices, or H2+ LCAO visualizations.",
    )
    parser.add_argument("--model", default="schrodingers_equation_hydrogen.pt", help="Path to trained model weights")
    parser.add_argument(
        "--out",
        default=None,
        help="Output path or template (use {n}, {l}, {m}, {orbital})",
    )
    parser.add_argument(
        "--r_max",
        type=float,
        default=6.0,
        help="Spatial extent for the grid in atomic units (half-size for H2 modes).",
    )
    parser.add_argument("--grid", type=int, default=None, help="Grid resolution per axis")
    parser.add_argument("--z", type=float, default=0.0, help="z-plane to slice (a.u.)")
    parser.add_argument(
        "--states",
        type=str,
        default="1,2",
        help="Comma-separated principal quantum numbers to visualise (e.g., 1,2)",
    )
    parser.add_argument(
        "--model_max_energy",
        type=int,
        default=None,
        help=(
            "Max principal quantum number the model was trained with (affects input normalization). "
            "Set this if you're rendering a subset of states from a model trained with a larger n."
        ),
    )
    parser.add_argument("--ls", type=str, default="0", help="Comma-separated l values (e.g., 0,1)")
    parser.add_argument("--ms", type=str, default="0", help="Comma-separated m values (e.g., -1,0,1)")
    parser.add_argument("--all", dest="all", action="store_true", help="Render all valid (l, m) combos for each n")
    parser.add_argument("--no-all", dest="all", action="store_false", help="Only render specified l/m values")
    parser.set_defaults(all=None)
    parser.add_argument(
        "--frames",
        type=int,
        default=None,
        help="Frames per orbital video (overrides --duration)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=3.0,
        help="Video duration in seconds per orbital",
    )
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate")
    parser.add_argument("--dpi", type=int, default=160, help="Video DPI")
    parser.add_argument("--iso_quantile", type=float, default=0.9999, help="Quantile threshold for the isosurface")
    parser.add_argument("--iso_value", type=float, default=None, help="Absolute density threshold for the isosurface")
    parser.add_argument("--max_points", type=int, default=120_000, help="Maximum points to render in scatter")
    parser.add_argument("--point_size", type=float, default=0.7, help="Scatter point size")
    parser.add_argument("--alpha", type=float, default=0.85, help="Scatter alpha")
    parser.add_argument("--cmap", type=str, default="magma", help="Matplotlib colormap for density")
    parser.add_argument("--elev", type=float, default=20.0, help="Base camera elevation angle")
    parser.add_argument("--elev_amp", type=float, default=10.0, help="Camera elevation oscillation amplitude")
    parser.add_argument("--spin", type=float, default=360.0, help="Total azimuthal spin in degrees")
    parser.add_argument("--batch", type=int, default=200_000, help="Batch size for model evaluation")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for point downsampling")
    parser.add_argument("--show_axes", action="store_true", help="Show 3D axes in orbital render")
    parser.add_argument(
        "--compare_video",
        action="store_true",
        help="Render orbitals video with side-by-side model/analytic/error panels",
    )
    parser.add_argument(
        "--harmonics",
        choices=("complex", "real"),
        default=None,
        help="Analytic spherical-harmonic convention (default: infer from model, else 'complex')",
    )
    parser.add_argument(
        "--print_norm",
        action="store_true",
        help="Print Monte Carlo estimate of ∫|psi|^2 dV inside a ball (per plotted state).",
    )
    parser.add_argument(
        "--renormalize_model",
        action="store_true",
        help="Scale model density by the estimated normalization integral before comparison (diagnostic).",
    )
    parser.add_argument("--norm_rmax", type=float, default=10.0, help="Radius for normalization estimate (a.u.)")
    parser.add_argument("--norm_samples", type=int, default=200_000, help="Samples for normalization estimate")
    parser.add_argument(
        "--real_harmonics",
        action="store_true",
        help="Deprecated: use --harmonics real",
    )
    parser.add_argument(
        "--h2_state",
        choices=("bonding", "antibonding", "both"),
        default="bonding",
        help="H2+ LCAO state to render (bonding, antibonding, or both).",
    )
    parser.add_argument(
        "--h2_axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis along which the H2+ nuclei are separated.",
    )
    parser.add_argument(
        "--h2_distance",
        type=float,
        default=2.0,
        help="Internuclear separation R for H2+ (a.u.).",
    )
    parser.add_argument(
        "--overlap_samples",
        type=int,
        default=200_000,
        help="Monte Carlo samples for H2+ overlap estimation (model).",
    )
    parser.add_argument(
        "--overlap_batch",
        type=int,
        default=200_000,
        help="Batch size for H2+ overlap estimation.",
    )
    args = parser.parse_args()
    if args.frames is None:
        if args.duration <= 0:
            raise ValueError("--duration must be positive")
        args.frames = max(1, int(round(args.duration * args.fps)))
    if args.out is None:
        if args.mode == "orbitals":
            args.out = "assets/hydrogen_{orbital}_compare.mp4" if args.compare_video else "assets/hydrogen_{orbital}.mp4"
        elif args.mode == "compare":
            args.out = "assets/hydrogen_{orbital}_compare.png"
        elif args.mode == "h2":
            args.out = "assets/h2_{orbital}_compare.mp4" if args.compare_video else "assets/h2_{orbital}.mp4"
        else:
            args.out = "assets/h2_{orbital}_compare.png"
    if args.grid is None:
        args.grid = 80 if args.mode in ("orbitals", "h2") else 220
    device = get_device()

    if args.mode in ("h2", "h2_compare"):
        max_energy = args.model_max_energy if args.model_max_energy is not None else 1
        if max_energy < 1:
            raise ValueError("--model_max_energy must be >= 1 for H2+ visualization.")
        model = load_model(args.model, device, max_energy=max_energy)

        if args.h2_state == "both":
            h2_states = ["bonding", "antibonding"]
        else:
            h2_states = [args.h2_state]

        for state in h2_states:
            if args.mode == "h2_compare":
                output = render_h2_comparison(
                    model_path=args.model,
                    output_path=args.out,
                    extent=args.r_max,
                    grid_points=args.grid,
                    z0=args.z,
                    axis=args.h2_axis,
                    nuclei_distance=args.h2_distance,
                    state=state,
                    model=model,
                    device=device,
                    model_max_energy=max_energy,
                    overlap_samples=args.overlap_samples,
                    overlap_batch=args.overlap_batch,
                )
                print(f"Saved H2+ comparison image for {state} to {output}")
            else:
                output = resolve_h2_output_path_with_suffix(
                    args.out,
                    state=state,
                    axis=args.h2_axis,
                    nuclei_distance=args.h2_distance,
                    default_suffix=".mp4",
                )
                if args.compare_video:
                    output = render_h2_orbital_comparison_video(
                        model=model,
                        device=device,
                        output_path=output,
                        extent=args.r_max,
                        grid_points=args.grid,
                        axis=args.h2_axis,
                        nuclei_distance=args.h2_distance,
                        state=state,
                        frames=args.frames,
                        fps=args.fps,
                        dpi=args.dpi,
                        batch_size=args.batch,
                        iso_quantile=args.iso_quantile,
                        iso_value=args.iso_value,
                        max_points=args.max_points,
                        seed=args.seed,
                        point_size=args.point_size,
                        alpha=args.alpha,
                        cmap=args.cmap,
                        elev=args.elev,
                        elev_amplitude=args.elev_amp,
                        spin_degrees=args.spin,
                        show_axes=args.show_axes,
                        overlap_samples=args.overlap_samples,
                        overlap_batch=args.overlap_batch,
                    )
                    print(f"Saved H2+ comparison video for {state} to {output}")
                else:
                    output = render_h2_orbital_video(
                        model=model,
                        device=device,
                        output_path=output,
                        extent=args.r_max,
                        grid_points=args.grid,
                        axis=args.h2_axis,
                        nuclei_distance=args.h2_distance,
                        state=state,
                        frames=args.frames,
                        fps=args.fps,
                        dpi=args.dpi,
                        batch_size=args.batch,
                        iso_quantile=args.iso_quantile,
                        iso_value=args.iso_value,
                        max_points=args.max_points,
                        seed=args.seed,
                        point_size=args.point_size,
                        alpha=args.alpha,
                        cmap=args.cmap,
                        elev=args.elev,
                        elev_amplitude=args.elev_amp,
                        spin_degrees=args.spin,
                        show_axes=args.show_axes,
                        overlap_samples=args.overlap_samples,
                        overlap_batch=args.overlap_batch,
                    )
                    print(f"Saved H2+ orbital video for {state} to {output}")
        return

    states = parse_int_list(args.states)
    ls = parse_int_list(args.ls)
    ms = parse_int_list(args.ms)
    if not states:
        raise ValueError("At least one state must be specified via --states")
    if args.all is None:
        args.all = True if args.mode == "orbitals" else False

    max_energy = max(states)
    model_max_energy = args.model_max_energy if args.model_max_energy is not None else max_energy
    if model_max_energy < max_energy:
        raise ValueError(
            f"--model_max_energy ({model_max_energy}) must be >= max(--states) ({max_energy})."
        )
    model = load_model(args.model, device, max_energy=model_max_energy)

    for n_value, l_value, m_value in iter_state_triplets(states, ls, ms, args.all):
        if args.mode == "compare":
            output = render_comparison(
                model_path=args.model,
                output_path=args.out,
                r_max=args.r_max,
                grid_points=args.grid,
                z0=args.z,
                n_value=n_value,
                l_value=l_value,
                m_value=m_value,
                harmonic_convention=("real" if args.real_harmonics else args.harmonics),
                model=model,
                device=device,
                model_max_energy=model_max_energy,
                renormalize_model=args.renormalize_model,
                norm_rmax=args.norm_rmax,
                norm_samples=args.norm_samples,
                print_norm=args.print_norm,
            )
            print(f"Saved comparison image for n={n_value}, l={l_value}, m={m_value} to {output}")
        else:
            output = resolve_output_path_with_suffix(args.out, n_value, l_value, m_value, default_suffix=".mp4")
            if args.compare_video:
                output = render_orbital_comparison_video(
                    model=model,
                    device=device,
                    output_path=output,
                    r_max=args.r_max,
                    grid_points=args.grid,
                    n_value=n_value,
                    l_value=l_value,
                    m_value=m_value,
                    frames=args.frames,
                    fps=args.fps,
                    dpi=args.dpi,
                    batch_size=args.batch,
                    iso_quantile=args.iso_quantile,
                    iso_value=args.iso_value,
                    max_points=args.max_points,
                    seed=args.seed,
                    point_size=args.point_size,
                    alpha=args.alpha,
                    cmap=args.cmap,
                    elev=args.elev,
                    elev_amplitude=args.elev_amp,
                    spin_degrees=args.spin,
                    show_axes=args.show_axes,
                    harmonic_convention=("real" if args.real_harmonics else args.harmonics),
                )
                print(f"Saved comparison video for n={n_value}, l={l_value}, m={m_value} to {output}")
            else:
                output = render_orbital_video(
                    model=model,
                    device=device,
                    output_path=output,
                    r_max=args.r_max,
                    grid_points=args.grid,
                    n_value=n_value,
                    l_value=l_value,
                    m_value=m_value,
                    frames=args.frames,
                    fps=args.fps,
                    dpi=args.dpi,
                    batch_size=args.batch,
                    iso_quantile=args.iso_quantile,
                    iso_value=args.iso_value,
                    max_points=args.max_points,
                    seed=args.seed,
                    point_size=args.point_size,
                    alpha=args.alpha,
                    cmap=args.cmap,
                    elev=args.elev,
                    elev_amplitude=args.elev_amp,
                    spin_degrees=args.spin,
                    show_axes=args.show_axes,
                )
                print(f"Saved orbital video for n={n_value}, l={l_value}, m={m_value} to {output}")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    main()
