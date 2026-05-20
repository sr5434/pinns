import argparse
import hashlib
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.animation import FFMpegWriter, FuncAnimation
from torch.quasirandom import SobolEngine

from schrodingers_equation_h2 import (
    H2_CHECKPOINT_PATTERN,
    H2_MODEL_DIR,
    SchrodingerEquationH2Ground,
    discover_h2_checkpoints,
    format_h2_distance_tag,
    get_existing_h2_checkpoint_path,
)


class LegacySchrodingerEquationH2Ground(torch.nn.Module):
    """Compatibility model for older checkpoints with a 5-feature core input."""

    def __init__(self, hidden=256):
        super().__init__()
        self.core = torch.nn.Sequential(
            torch.nn.Linear(5, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden, hidden // 2),
            torch.nn.Tanh(),
            torch.nn.Linear(hidden // 2, 1),
        )

    def _features(self, r1A, r1B, r2A, r2B, r12):
        s1 = r1A + r2A
        s2 = r1B + r2B
        d1 = torch.abs(r1A - r2A)
        d2 = torch.abs(r1B - r2B)
        return torch.cat([s1, s2, d1, d2, r12], dim=-1)

    def forward(self, r1A, r1B, r2A, r2B, r12):
        f12 = self.core(self._features(r1A, r1B, r2A, r2B, r12))
        f21 = self.core(self._features(r2A, r2B, r1A, r1B, r12))
        base = 0.5 * (f12 + f21)
        envelope = torch.exp(-0.8 * (r1A + r1B + r2A + r2B))
        return base * envelope

    @staticmethod
    def pair_distances_from_collocation_points(collocation_points, nuclei_distance, axis="x", eps=1e-6, r12_floor=1e-3):
        if collocation_points.ndim == 2 and collocation_points.shape[1] == 6:
            points = collocation_points.view(-1, 2, 3)
        elif collocation_points.ndim == 3 and collocation_points.shape[1:] == (2, 3):
            points = collocation_points
        else:
            raise ValueError("collocation_points must have shape (N, 6) or (N, 2, 3)")

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

        def safe_distance(a, b):
            diff = a - b
            return torch.sqrt(torch.sum(diff * diff, dim=-1, keepdim=True) + eps ** 2)

        r1A = safe_distance(electron_1, nucleus_a)
        r1B = safe_distance(electron_1, nucleus_b)
        r2A = safe_distance(electron_2, nucleus_a)
        r2B = safe_distance(electron_2, nucleus_b)
        r12 = safe_distance(electron_1, electron_2)
        r12_eff = torch.sqrt(r12 ** 2 + r12_floor ** 2)
        return r1A, r1B, r2A, r2B, r12_eff


def get_device(prefer_mps=True, prefer_cuda=True):
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_nuclei_positions(axis, nuclei_distance, device, dtype=torch.float32):
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


def analytic_h2_overlap_1s(nuclei_distance):
    r = float(nuclei_distance)
    return np.exp(-r) * (1.0 + r + (r ** 2) / 3.0)


def normalize_density_grid_to_two_electrons(density_grid, coords, eps=1e-12):
    if coords.size < 2:
        raise ValueError("coords must contain at least 2 points for volume integration.")
    spacing = float(coords[1] - coords[0])
    dV = spacing ** 3
    integral = float(np.sum(density_grid, dtype=np.float64) * dV)
    if integral <= eps:
        raise ValueError("Baseline density integral is non-positive; cannot normalize.")
    scale = 2.0 / integral
    return density_grid * scale, integral, scale


def _require_pyscf():
    try:
        from pyscf import fci  # noqa: F401
        from pyscf import gto  # noqa: F401
        from pyscf import scf  # noqa: F401
        from pyscf.dft import numint  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "PySCF is required for '--baseline hf' and '--baseline fci' "
            "(and for '--baseline reference' fallback when reference data is missing). "
            "Install it with: pip install pyscf"
        ) from exc
    return gto, scf, fci, numint


def _h2_atom_string(axis, nuclei_distance):
    half = 0.5 * float(nuclei_distance)
    if axis == "x":
        return f"H {-half} 0.0 0.0; H {half} 0.0 0.0"
    if axis == "y":
        return f"H 0.0 {-half} 0.0; H 0.0 {half} 0.0"
    if axis == "z":
        return f"H 0.0 0.0 {-half}; H 0.0 0.0 {half}"
    raise ValueError("axis must be 'x', 'y', or 'z'")


def _build_h2_mol(axis, nuclei_distance, basis, charge, spin, gto_module):
    mol = gto_module.M(
        atom=_h2_atom_string(axis, nuclei_distance),
        basis=basis,
        charge=int(charge),
        spin=int(spin),
        unit="Bohr",
        verbose=0,
    )
    return mol


def _compute_abinitio_density_on_points(points_e1_np, axis, nuclei_distance, basis, charge, spin, method):
    gto, scf, fci, numint = _require_pyscf()
    method_key = method.lower().strip()
    if method_key not in {"hf", "fci"}:
        raise ValueError("method must be 'hf' or 'fci'.")

    mol = _build_h2_mol(
        axis=axis,
        nuclei_distance=nuclei_distance,
        basis=basis,
        charge=charge,
        spin=spin,
        gto_module=gto,
    )
    mf = scf.RHF(mol)
    mf.verbose = 0
    hf_energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("PySCF RHF did not converge for H2 baseline computation.")

    if method_key == "hf":
        dm_ao = mf.make_rdm1()
        metadata = {
            "method": "hf",
            "basis": basis,
            "energy_hartree": hf_energy,
            "density_source": "PySCF RHF density matrix",
        }
    else:
        cisolver = fci.FCI(mol, mf.mo_coeff)
        cisolver.verbose = 0
        fci_energy, fcivec = cisolver.kernel()
        if fcivec is None:
            raise RuntimeError("PySCF FCI solver returned no wavefunction.")
        nmo = int(mf.mo_coeff.shape[1])
        dm1_mo = cisolver.make_rdm1(fcivec, nmo, mol.nelectron)
        dm_ao = mf.mo_coeff @ dm1_mo @ mf.mo_coeff.T
        metadata = {
            "method": "fci",
            "basis": basis,
            "energy_hartree": float(fci_energy),
            "hf_energy_hartree": hf_energy,
            "density_source": "PySCF FCI 1-RDM",
        }

    ao = numint.eval_ao(mol, points_e1_np)
    rho = numint.eval_rho(mol, ao, dm_ao)
    rho = np.asarray(rho, dtype=np.float64).reshape(-1).astype(np.float32, copy=False)
    return rho, metadata


def _safe_float_match(a, b, tol=1e-6):
    try:
        return abs(float(a) - float(b)) <= tol
    except (TypeError, ValueError):
        return False


def load_reference_density_from_manifest(
    reference_dir,
    axis,
    nuclei_distance,
    r_max,
    grid_points,
    abinit_basis,
    preferred_dtype="float32",
):
    reference_dir = pathlib.Path(reference_dir)
    manifest_path = reference_dir / "manifest.json"
    if not manifest_path.exists():
        return None, None

    manifest = json.loads(manifest_path.read_text())
    entries = manifest.get("entries", [])
    if not isinstance(entries, list):
        raise ValueError(f"Invalid manifest format in {manifest_path}: 'entries' must be a list.")

    candidates = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        if entry.get("kind") != "density_grid":
            continue
        if str(entry.get("axis", "")).lower() != str(axis).lower():
            continue
        if int(entry.get("grid", -1)) != int(grid_points):
            continue
        if not _safe_float_match(entry.get("R"), nuclei_distance):
            continue
        if not _safe_float_match(entry.get("r_max"), r_max):
            continue
        candidates.append(entry)

    if not candidates:
        return None, None

    def sort_key(entry):
        method = str(entry.get("method", "")).lower()
        method_rank = {"fci": 0, "hf": 1}.get(method, 2)
        basis = str(entry.get("basis", ""))
        basis_rank = 0 if basis == str(abinit_basis) else 1
        return (method_rank, basis_rank, str(entry.get("id", "")))

    selected = sorted(candidates, key=sort_key)[0]
    file_name = selected.get("file")
    if not file_name:
        raise ValueError(f"Reference entry missing 'file' field in {manifest_path}.")
    data_path = reference_dir / file_name
    if not data_path.exists():
        raise FileNotFoundError(f"Reference density file not found: {data_path}")

    expected_sha = selected.get("sha256")
    if expected_sha:
        digest = hashlib.sha256(data_path.read_bytes()).hexdigest()
        if digest.lower() != str(expected_sha).lower():
            raise ValueError(
                f"SHA256 mismatch for {data_path}. expected={expected_sha}, actual={digest}"
            )

    with np.load(data_path) as data:
        if "density" in data:
            density = data["density"]
        elif len(data.files) == 1:
            density = data[data.files[0]]
        else:
            raise ValueError(f"Reference file {data_path} does not contain a 'density' array.")

    expected_shape = (int(grid_points), int(grid_points), int(grid_points))
    if tuple(density.shape) != expected_shape:
        raise ValueError(
            f"Reference density shape mismatch in {data_path}: "
            f"expected {expected_shape}, got {tuple(density.shape)}"
        )

    dtype_key = str(preferred_dtype).lower()
    if dtype_key == "float16":
        density = density.astype(np.float16, copy=False).astype(np.float32, copy=False)
    else:
        density = density.astype(np.float32, copy=False)

    metadata = {
        "method": selected.get("method", "reference"),
        "id": selected.get("id", ""),
        "basis": selected.get("basis", ""),
        "source": selected.get("source", ""),
        "citation": selected.get("citation", ""),
        "energy_hartree": selected.get("energy_hartree"),
        "file": str(data_path),
    }
    return density, metadata


def compute_baseline_density_grid(
    baseline,
    points_e1,
    coords,
    grid_points,
    axis,
    nuclei_distance,
    abinit_basis="cc-pVTZ",
    abinit_charge=0,
    abinit_spin=0,
    reference_dir="assets/reference_data/h2",
    reference_strict=False,
    reference_dtype="float32",
):
    baseline_key = str(baseline).lower().strip()
    points_e1_cpu = points_e1.detach().to("cpu")
    metadata = {}

    if baseline_key == "lcao1s":
        nucleus_a, nucleus_b = get_nuclei_positions(axis, nuclei_distance, device=points_e1.device, dtype=torch.float32)
        r1A = torch.linalg.norm(points_e1 - nucleus_a.view(1, 3), dim=-1).clamp_min(1e-6)
        r1B = torch.linalg.norm(points_e1 - nucleus_b.view(1, 3), dim=-1).clamp_min(1e-6)
        overlap = analytic_h2_overlap_1s(nuclei_distance)
        norm_lcao = np.sqrt(2.0 * max(1e-8, 1.0 + overlap))
        phi_a = torch.exp(-r1A) / np.sqrt(np.pi)
        phi_b = torch.exp(-r1B) / np.sqrt(np.pi)
        phi_g = (phi_a + phi_b) / norm_lcao
        density = (2.0 * phi_g.pow(2)).reshape(-1).detach().cpu().numpy().astype(np.float32, copy=False)
        label = "LCAO-1s marginal density"
        metadata["method"] = "lcao1s"

    elif baseline_key in {"hf", "fci"}:
        density, metadata = _compute_abinitio_density_on_points(
            points_e1_np=points_e1_cpu.numpy().astype(np.float64, copy=False),
            axis=axis,
            nuclei_distance=nuclei_distance,
            basis=abinit_basis,
            charge=abinit_charge,
            spin=abinit_spin,
            method=baseline_key,
        )
        label = f"{baseline_key.upper()} marginal density"

    elif baseline_key == "reference":
        density, metadata = load_reference_density_from_manifest(
            reference_dir=reference_dir,
            axis=axis,
            nuclei_distance=nuclei_distance,
            r_max=float(np.max(np.abs(coords))),
            grid_points=grid_points,
            abinit_basis=abinit_basis,
            preferred_dtype=reference_dtype,
        )
        if density is None:
            if reference_strict:
                raise FileNotFoundError(
                    "No matching reference density was found in manifest and --reference_strict is set."
                )
            print("Reference baseline not found. Falling back to FCI baseline.")
            density, metadata = _compute_abinitio_density_on_points(
                points_e1_np=points_e1_cpu.numpy().astype(np.float64, copy=False),
                axis=axis,
                nuclei_distance=nuclei_distance,
                basis=abinit_basis,
                charge=abinit_charge,
                spin=abinit_spin,
                method="fci",
            )
            label = "FCI marginal density (reference fallback)"
            metadata["fallback_from"] = "reference"
        else:
            label = "Reference marginal density"
    else:
        raise ValueError(f"Unsupported baseline '{baseline}'.")

    density_grid = density.reshape(grid_points, grid_points, grid_points)
    density_grid, integral_before, scale = normalize_density_grid_to_two_electrons(density_grid, coords)
    metadata["integral_before_norm"] = integral_before
    metadata["norm_scale"] = scale
    return density_grid.astype(np.float32, copy=False), label, metadata


def load_model(model_path, device):
    path = pathlib.Path(model_path)
    if not path.exists():
        alt = pathlib.Path("schrodingers_equation") / path.name
        if alt.exists():
            path = alt
        else:
            raise FileNotFoundError(f"Checkpoint not found: {model_path}")

    state_dict = torch.load(path, map_location=device)
    if "core.0.weight" not in state_dict:
        raise ValueError("Unsupported checkpoint format: missing core.0.weight")

    input_features = int(state_dict["core.0.weight"].shape[1])
    if input_features == 4:
        model = SchrodingerEquationH2Ground().to(device)
    elif input_features == 5:
        model = LegacySchrodingerEquationH2Ground().to(device)
    else:
        raise ValueError(
            f"Unsupported checkpoint architecture with core.0.weight second dim={input_features}"
        )

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise ValueError(f"Unexpected keys in checkpoint: {unexpected}")
    if input_features == 4:
        allowed_missing = {
            "b",
            "covalent_weight",
            "ionic_weight",
            "three_body_radial_scale",
            "three_body_ee_scale",
        }
        if not set(missing).issubset(allowed_missing):
            raise ValueError(f"Missing required keys in checkpoint: {missing}")
    elif missing:
        raise ValueError(f"Missing required keys in checkpoint: {missing}")

    model.eval()
    return model


def compute_h2_marginal_densities_3d(
    model,
    device,
    r_max,
    grid_points,
    axis,
    nuclei_distance,
    baseline="lcao1s",
    abinit_basis="cc-pVTZ",
    abinit_charge=0,
    abinit_spin=0,
    reference_dir="assets/reference_data/h2",
    reference_strict=False,
    reference_dtype="float32",
    normalize_model_density=False,
    r2_samples=256,
    r2_box=8.0,
    e1_batch_size=128,
    pair_batch_size=262_144,
):
    coords = torch.linspace(-r_max, r_max, grid_points, dtype=torch.float32, device=device)
    xx, yy, zz = torch.meshgrid(coords, coords, coords, indexing="ij")
    points_e1 = torch.stack([xx, yy, zz], dim=-1).reshape(-1, 3)

    n_points = points_e1.shape[0]
    model_integral = torch.empty(n_points, device="cpu", dtype=torch.float32)

    sobol = SobolEngine(dimension=3, scramble=True, seed=7)
    points_e2 = (r2_box * sobol.draw(r2_samples, dtype=torch.float32) - (r2_box / 2.0)).to(device)
    e2_volume = float(r2_box) ** 3

    with torch.no_grad():
        for start in range(0, n_points, e1_batch_size):
            e1_batch = points_e1[start : start + e1_batch_size]
            n_e1 = e1_batch.shape[0]
            integral_sum = torch.zeros(n_e1, device=device, dtype=torch.float32)

            e2_start = 0
            while e2_start < r2_samples:
                max_chunk = max(1, pair_batch_size // max(1, n_e1))
                e2_chunk_n = min(max_chunk, r2_samples - e2_start)
                e2_chunk = points_e2[e2_start : e2_start + e2_chunk_n]

                e1_rep = e1_batch[:, None, :].expand(n_e1, e2_chunk_n, 3)
                e2_rep = e2_chunk[None, :, :].expand(n_e1, e2_chunk_n, 3)
                pairs = torch.cat([e1_rep, e2_rep], dim=-1).reshape(-1, 6)

                r1A, r1B, r2A, r2B, r12 = model.pair_distances_from_collocation_points(
                    pairs,
                    nuclei_distance=nuclei_distance,
                    axis=axis,
                )
                psi, _ = model(r1A, r1B, r2A, r2B, r12)
                density = psi.squeeze(-1).pow(2).reshape(n_e1, e2_chunk_n)
                integral_sum += density.sum(dim=1)
                e2_start += e2_chunk_n

            model_integral[start : start + n_e1] = (integral_sum * (e2_volume / float(r2_samples))).cpu()

    coords_np = coords.cpu().numpy()
    baseline_grid, baseline_label, baseline_metadata = compute_baseline_density_grid(
        baseline=baseline,
        points_e1=points_e1,
        coords=coords_np,
        grid_points=grid_points,
        axis=axis,
        nuclei_distance=nuclei_distance,
        abinit_basis=abinit_basis,
        abinit_charge=abinit_charge,
        abinit_spin=abinit_spin,
        reference_dir=reference_dir,
        reference_strict=reference_strict,
        reference_dtype=reference_dtype,
    )

    model_grid = model_integral.view(grid_points, grid_points, grid_points).numpy()
    if coords_np.size < 2:
        raise ValueError("Need at least 2 grid points per axis for density integration.")
    spacing = float(coords_np[1] - coords_np[0])
    dV = spacing ** 3
    model_integral_raw = float(np.sum(model_grid, dtype=np.float64) * dV)

    model_metadata = {
        "integral_raw": model_integral_raw,
        "normalized_to_two_electrons": bool(normalize_model_density),
    }
    if normalize_model_density:
        model_grid, integral_before_norm, model_scale = normalize_density_grid_to_two_electrons(
            model_grid,
            coords_np,
        )
        model_metadata["integral_before_norm"] = integral_before_norm
        model_metadata["norm_scale"] = model_scale
        model_integral_after = float(np.sum(model_grid, dtype=np.float64) * dV)
        model_metadata["integral_after_norm"] = model_integral_after

    return coords_np, model_grid, baseline_grid, baseline_label, baseline_metadata, model_metadata


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
    r2 = coord2[:, None, None] + coord2[None, :, None] + coord2[None, None, :]
    sphere_mask = r2 <= (r_max ** 2 + 1e-6)
    valid_density = density_grid[sphere_mask]
    if valid_density.size == 0:
        raise ValueError("No density samples inside r_max; increase --r_max or --grid.")

    if iso_value is None:
        threshold = float(np.quantile(valid_density, iso_quantile))
    else:
        threshold = float(iso_value)

    mask = (density_grid >= threshold) & sphere_mask
    indices = np.argwhere(mask)
    if indices.size == 0:
        raise ValueError("No points above threshold. Lower --iso_quantile or set --iso_value.")

    if max_points and indices.shape[0] > max_points:
        rng = np.random.default_rng(seed)
        selected = rng.choice(indices.shape[0], size=max_points, replace=False)
        indices = indices[selected]

    x = coords[indices[:, 0]]
    y = coords[indices[:, 1]]
    z = coords[indices[:, 2]]
    values = density_grid[indices[:, 0], indices[:, 1], indices[:, 2]]
    return x, y, z, values, threshold


def add_nuclei_markers_3d(ax, axis, nuclei_distance):
    nucleus_a, nucleus_b = get_nuclei_positions(axis, nuclei_distance, device="cpu", dtype=torch.float32)
    nucleus_a = nucleus_a.numpy()
    nucleus_b = nucleus_b.numpy()
    ax.scatter(
        [nucleus_a[0], nucleus_b[0]],
        [nucleus_a[1], nucleus_b[1]],
        [nucleus_a[2], nucleus_b[2]],
        c="#4fc3f7",
        s=28,
        marker="o",
        linewidths=0.4,
        edgecolors="white",
        depthshade=False,
    )


def render_h2_marginal_comparison_video(
    model,
    device,
    output_path,
    r_max,
    grid_points,
    axis,
    nuclei_distance,
    baseline="lcao1s",
    abinit_basis="cc-pVTZ",
    abinit_charge=0,
    abinit_spin=0,
    reference_dir="assets/reference_data/h2",
    reference_strict=False,
    reference_dtype="float32",
    normalize_model_density=False,
    frames=180,
    fps=30,
    dpi=160,
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
    show_markers=True,
    shared_density_scale=True,
    r2_samples=256,
    r2_box=8.0,
    e1_batch_size=128,
    pair_batch_size=262_144,
):
    coords, model_density_grid, baseline_density_grid, baseline_label, baseline_metadata, model_metadata = compute_h2_marginal_densities_3d(
        model=model,
        device=device,
        r_max=r_max,
        grid_points=grid_points,
        axis=axis,
        nuclei_distance=nuclei_distance,
        baseline=baseline,
        abinit_basis=abinit_basis,
        abinit_charge=abinit_charge,
        abinit_spin=abinit_spin,
        reference_dir=reference_dir,
        reference_strict=reference_strict,
        reference_dtype=reference_dtype,
        normalize_model_density=normalize_model_density,
        r2_samples=r2_samples,
        r2_box=r2_box,
        e1_batch_size=e1_batch_size,
        pair_batch_size=pair_batch_size,
    )

    error_grid = np.abs(model_density_grid - baseline_density_grid)
    mae = float(error_grid.mean())

    model_x, model_y, model_z, model_values, model_threshold = sample_orbital_points(
        coords=coords,
        density_grid=model_density_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )
    baseline_x, baseline_y, baseline_z, baseline_values, baseline_threshold = sample_orbital_points(
        coords=coords,
        density_grid=baseline_density_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=iso_value,
        max_points=max_points,
        seed=seed,
    )
    error_x, error_y, error_z, error_values, error_threshold = sample_orbital_points(
        coords=coords,
        density_grid=error_grid,
        r_max=r_max,
        iso_quantile=iso_quantile,
        iso_value=None,
        max_points=max_points,
        seed=seed,
    )

    if shared_density_scale:
        density_vmax = float(max(model_density_grid.max(), baseline_density_grid.max()))
        model_vmax = density_vmax
        baseline_vmax = density_vmax
    else:
        model_vmax = float(model_density_grid.max())
        baseline_vmax = float(baseline_density_grid.max())

    if model_vmax <= 0.0:
        model_vmax = 1e-12
    if baseline_vmax <= 0.0:
        baseline_vmax = 1e-12

    error_vmax = float(error_grid.max())
    if error_vmax <= 0.0:
        error_vmax = 1e-12

    print(
        "Rendering H2 marginal comparison video | "
        f"model pts={model_x.shape[0]} (thr={model_threshold:.3e}), "
        f"baseline pts={baseline_x.shape[0]} (thr={baseline_threshold:.3e}), "
        f"error pts={error_x.shape[0]} (thr={error_threshold:.3e}), "
        f"baseline={baseline}, r2_samples={r2_samples}, r2_box={r2_box:g}"
    )
    if baseline_metadata:
        meta_parts = []
        for key in ("method", "basis", "energy_hartree", "source", "file", "fallback_from"):
            if key in baseline_metadata and baseline_metadata[key] not in (None, ""):
                meta_parts.append(f"{key}={baseline_metadata[key]}")
        if meta_parts:
            print("Baseline metadata | " + ", ".join(meta_parts))
    if model_metadata:
        model_meta_parts = []
        for key in ("integral_raw", "integral_before_norm", "integral_after_norm", "norm_scale"):
            if key in model_metadata and model_metadata[key] not in (None, ""):
                model_meta_parts.append(f"{key}={model_metadata[key]}")
        model_meta_parts.append(f"normalized_to_two_electrons={bool(model_metadata.get('normalized_to_two_electrons', False))}")
        print("Model metadata | " + ", ".join(model_meta_parts))

    fig = plt.figure(figsize=(15, 5))
    fig.patch.set_facecolor("black")
    axes = [
        fig.add_subplot(1, 3, 1, projection="3d"),
        fig.add_subplot(1, 3, 2, projection="3d"),
        fig.add_subplot(1, 3, 3, projection="3d"),
    ]
    model_title = "Model marginal density (norm=2)" if normalize_model_density else "Model marginal density"
    titles = [model_title, baseline_label, "Absolute error"]

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
            for axis_obj in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis_obj.set_tick_params(colors="white")
            ax.xaxis.label.set_color("white")
            ax.yaxis.label.set_color("white")
            ax.zaxis.label.set_color("white")
        if show_markers:
            add_nuclei_markers_3d(ax, axis=axis, nuclei_distance=nuclei_distance)

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
        vmax=model_vmax,
    )
    sc_baseline = axes[1].scatter(
        baseline_x,
        baseline_y,
        baseline_z,
        c=baseline_values,
        s=point_size,
        cmap=cmap,
        alpha=alpha,
        linewidths=0.0,
        vmin=0.0,
        vmax=baseline_vmax,
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

    fig.suptitle(
        f"H2 one-electron marginal density | R={nuclei_distance:g}, axis={axis}, integrated over e2",
        color="white",
    )
    fig.text(0.5, 0.02, f"MAE: {mae:.6e}", ha="center", va="bottom", color="white", fontsize=10)

    def update(frame):
        azim = (spin_degrees * frame / max(1, frames)) % 360.0
        elev_current = elev + elev_amplitude * np.sin(2.0 * np.pi * frame / max(1, frames))
        for ax in axes:
            ax.view_init(elev=elev_current, azim=azim)
        return (sc_model, sc_baseline, sc_error)

    anim = FuncAnimation(fig, update, frames=frames, interval=1000 / fps, blit=False)

    output_path = pathlib.Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    writer = FFMpegWriter(fps=fps, codec="libx264", bitrate=1800)
    try:
        anim.save(output_path, writer=writer, dpi=dpi)
    except Exception as exc:
        plt.close(fig)
        raise RuntimeError("Failed to write video. Ensure ffmpeg is installed.") from exc
    plt.close(fig)
    return output_path


def infer_h2_distance_from_checkpoint_path(model_path):
    model_path = pathlib.Path(model_path)
    match = H2_CHECKPOINT_PATTERN.fullmatch(model_path.name)
    if match is None:
        raise ValueError(
            "Unable to infer H2 bond length from checkpoint name. "
            "Pass --h2_distance explicitly or use a checkpoint named like "
            "'schrodingers_equation_h2_ground_R1.4.pt'."
        )
    return float(match.group("R"))


def make_default_output_path(output_dir, baseline, nuclei_distance, axis):
    return pathlib.Path(output_dir) / (
        f"h2_ground_marginal_compare_{baseline}_R{format_h2_distance_tag(nuclei_distance)}_{axis}.mp4"
    )


def make_default_summary_path(output_dir, baseline, axis):
    return pathlib.Path(output_dir) / f"h2_ground_marginal_compare_{baseline}_{axis}_batch.json"


def resolve_render_jobs(args, parser):
    if args.model is not None:
        nuclei_distance = (
            float(args.h2_distance)
            if args.h2_distance is not None
            else infer_h2_distance_from_checkpoint_path(args.model)
        )
        output_path = (
            pathlib.Path(args.out)
            if args.out is not None
            else make_default_output_path(args.output_dir, args.baseline, nuclei_distance, args.h2_axis)
        )
        return [
            {
                "R": nuclei_distance,
                "checkpoint": pathlib.Path(args.model),
                "output": output_path,
            }
        ]

    if args.h2_distance is not None:
        model_path = get_existing_h2_checkpoint_path(args.h2_distance)
        if not pathlib.Path(model_path).exists():
            parser.error(f"No H2 checkpoint found for R={float(args.h2_distance):g}: {model_path}")
        output_path = (
            pathlib.Path(args.out)
            if args.out is not None
            else make_default_output_path(args.output_dir, args.baseline, args.h2_distance, args.h2_axis)
        )
        return [
            {
                "R": float(args.h2_distance),
                "checkpoint": pathlib.Path(model_path),
                "output": output_path,
            }
        ]

    checkpoints = discover_h2_checkpoints(args.models_dir)
    if not checkpoints:
        parser.error(f"No R-tagged H2 checkpoints were found in {args.models_dir}.")
    if args.out is not None:
        parser.error("--out can only be used for single-checkpoint renders. Use --output-dir for batch renders.")
    return [
        {
            "R": float(nuclei_distance),
            "checkpoint": pathlib.Path(model_path),
            "output": make_default_output_path(args.output_dir, args.baseline, nuclei_distance, args.h2_axis),
        }
        for nuclei_distance, model_path in checkpoints
    ]


def save_render_summary(summary_path, rows, args, resolved_device):
    payload = {
        "models_dir": str(pathlib.Path(args.models_dir)),
        "explicit_model": None if args.model is None else str(pathlib.Path(args.model)),
        "requested_h2_distance": None if args.h2_distance is None else float(args.h2_distance),
        "resolved_device": resolved_device,
        "baseline": args.baseline,
        "axis": args.h2_axis,
        "r_max": float(args.r_max),
        "grid": int(args.grid),
        "frames": int(args.frames),
        "fps": int(args.fps),
        "normalize_model_density": bool(args.normalize_model_density),
        "marginal_r2_samples": int(args.marginal_r2_samples),
        "marginal_r2_box": float(args.marginal_r2_box),
        "renders": rows,
    }
    summary_path = pathlib.Path(summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2))


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Neutral H2 visualization: render rotating 3-panel marginal-density comparison videos "
            "for one checkpoint or for every saved R-tagged checkpoint, using one-electron marginal "
            "density integrated over electron-2 and selectable baselines (LCAO/HF/FCI/reference)."
        )
    )
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "Explicit path to a trained H2 checkpoint. If omitted and --h2_distance is also omitted, "
            "the script renders every discovered R-tagged checkpoint under --models-dir."
        ),
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=H2_MODEL_DIR,
        help="Directory containing R-tagged H2 checkpoints for batch rendering.",
    )
    parser.add_argument(
        "--baseline",
        choices=("lcao1s", "hf", "fci", "reference"),
        default="lcao1s",
        help="Baseline density used in panel 2.",
    )
    parser.add_argument(
        "--abinit_basis",
        type=str,
        default="cc-pVTZ",
        help="PySCF basis set used for HF/FCI baselines and reference fallback.",
    )
    parser.add_argument(
        "--abinit_charge",
        type=int,
        default=0,
        help="Molecular charge for ab initio baseline computation.",
    )
    parser.add_argument(
        "--abinit_spin",
        type=int,
        default=0,
        help="2S spin value for ab initio baseline computation.",
    )
    parser.add_argument(
        "--reference_dir",
        type=str,
        default="assets/reference_data/h2",
        help="Directory containing local manifest.json and cached precomputed H2 reference data.",
    )
    parser.add_argument(
        "--reference_strict",
        action="store_true",
        help="When using --baseline reference, fail if no matching entry is found instead of falling back to FCI.",
    )
    parser.add_argument(
        "--reference_dtype",
        choices=("float16", "float32"),
        default="float32",
        help="Preferred dtype when loading reference density grids.",
    )
    parser.add_argument(
        "--normalize_model_density",
        action="store_true",
        help=(
            "Normalize model one-electron marginal density to 2 electrons on the plotted r1 grid "
            "for apples-to-apples intensity comparison with normalized baselines."
        ),
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output .mp4 path for single-checkpoint renders. Batch renders use --output-dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path("assets"),
        help="Directory used for default batch-render video paths.",
    )
    parser.add_argument(
        "--summary-json",
        type=pathlib.Path,
        default=None,
        help="Optional JSON path for batch render summary. Defaults to output-dir stem when rendering multiple checkpoints.",
    )
    parser.add_argument("--r_max", type=float, default=6.0, help="Spatial extent in a.u.")
    parser.add_argument("--grid", type=int, default=40, help="Grid resolution per axis for electron-1.")
    parser.add_argument("--fps", type=int, default=30, help="Video frame rate.")
    parser.add_argument("--duration", type=float, default=3.0, help="Video duration in seconds.")
    parser.add_argument("--frames", type=int, default=None, help="Frames per video (overrides --duration).")
    parser.add_argument("--dpi", type=int, default=160, help="Video DPI.")
    parser.add_argument("--iso_quantile", type=float, default=0.996, help="Isosurface quantile threshold.")
    parser.add_argument("--iso_value", type=float, default=None, help="Absolute isosurface threshold.")
    parser.add_argument("--max_points", type=int, default=120_000, help="Maximum points in scatter plot.")
    parser.add_argument("--point_size", type=float, default=0.6, help="3D scatter point size.")
    parser.add_argument("--alpha", type=float, default=0.85, help="Scatter alpha.")
    parser.add_argument("--cmap", type=str, default="magma", help="Colormap for model and baseline.")
    parser.add_argument("--error_cmap", type=str, default="viridis", help="Colormap for error panel.")
    parser.add_argument("--elev", type=float, default=20.0, help="Base camera elevation angle.")
    parser.add_argument("--elev_amp", type=float, default=10.0, help="Camera elevation oscillation amplitude.")
    parser.add_argument("--spin", type=float, default=360.0, help="Total camera azimuth spin in degrees.")
    parser.add_argument("--seed", type=int, default=7, help="Random seed used for point downsampling.")
    parser.add_argument("--show_axes", action="store_true", help="Show 3D axes.")
    parser.add_argument("--hide_markers", action="store_true", help="Hide proton markers.")
    parser.add_argument(
        "--h2_axis",
        choices=("x", "y", "z"),
        default="x",
        help="Axis along which nuclei are separated.",
    )
    parser.add_argument(
        "--h2_distance",
        type=float,
        default=None,
        help="Internuclear distance R in a.u. Restricts rendering to a single discovered checkpoint when --model is omitted.",
    )
    parser.add_argument(
        "--independent_density_scale",
        action="store_true",
        help="Use separate color scale maxima for model and baseline panels.",
    )
    parser.add_argument(
        "--marginal_r2_samples",
        type=int,
        default=256,
        help="Sobol samples used to integrate out electron-2.",
    )
    parser.add_argument(
        "--marginal_r2_box",
        type=float,
        default=8.0,
        help="Box length for electron-2 integration domain.",
    )
    parser.add_argument(
        "--marginal_e1_batch",
        type=int,
        default=128,
        help="Batch size for electron-1 grid points.",
    )
    parser.add_argument(
        "--marginal_pair_batch",
        type=int,
        default=262_144,
        help="Max flattened pair evaluations per forward pass.",
    )
    args = parser.parse_args()

    if args.frames is None:
        if args.duration <= 0:
            raise ValueError("--duration must be positive.")
        args.frames = max(1, int(round(args.duration * args.fps)))

    device = get_device()
    render_jobs = resolve_render_jobs(args, parser)
    render_rows = []

    for index, job in enumerate(render_jobs, start=1):
        checkpoint_path = pathlib.Path(job["checkpoint"])
        nuclei_distance = float(job["R"])
        output = pathlib.Path(job["output"])
        print(
            f"[{index}/{len(render_jobs)}] Rendering neutral H2 marginal comparison video "
            f"for R={nuclei_distance:g} from {checkpoint_path}"
        )
        model = load_model(checkpoint_path, device)
        output = render_h2_marginal_comparison_video(
            model=model,
            device=device,
            output_path=output,
            r_max=args.r_max,
            grid_points=args.grid,
            axis=args.h2_axis,
            nuclei_distance=nuclei_distance,
            baseline=args.baseline,
            abinit_basis=args.abinit_basis,
            abinit_charge=args.abinit_charge,
            abinit_spin=args.abinit_spin,
            reference_dir=args.reference_dir,
            reference_strict=args.reference_strict,
            reference_dtype=args.reference_dtype,
            normalize_model_density=args.normalize_model_density,
            frames=args.frames,
            fps=args.fps,
            dpi=args.dpi,
            iso_quantile=args.iso_quantile,
            iso_value=args.iso_value,
            max_points=args.max_points,
            seed=args.seed,
            point_size=args.point_size,
            alpha=args.alpha,
            cmap=args.cmap,
            error_cmap=args.error_cmap,
            elev=args.elev,
            elev_amplitude=args.elev_amp,
            spin_degrees=args.spin,
            show_axes=args.show_axes,
            show_markers=(not args.hide_markers),
            shared_density_scale=(not args.independent_density_scale),
            r2_samples=args.marginal_r2_samples,
            r2_box=args.marginal_r2_box,
            e1_batch_size=args.marginal_e1_batch,
            pair_batch_size=args.marginal_pair_batch,
        )
        render_rows.append(
            {
                "R": nuclei_distance,
                "checkpoint": str(checkpoint_path),
                "output": str(output),
            }
        )

    summary_path = None
    if args.summary_json is not None:
        summary_path = pathlib.Path(args.summary_json)
    elif len(render_jobs) > 1:
        summary_path = make_default_summary_path(args.output_dir, args.baseline, args.h2_axis)

    if summary_path is not None:
        save_render_summary(summary_path, render_rows, args, device)
        print(f"Saved H2 visualization batch summary to {summary_path}")

    print(
        f"Saved {len(render_rows)} neutral H2 marginal comparison video(s) "
        f"({args.baseline}, normalize_model_density={args.normalize_model_density})"
    )


if __name__ == "__main__":
    plt.switch_backend("Agg")
    main()
