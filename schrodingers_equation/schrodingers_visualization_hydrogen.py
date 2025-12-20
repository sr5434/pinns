import argparse
import pathlib
import torch
import matplotlib.pyplot as plt
from schrodingers_equation_hydrogen import SchrodingerEquationHydrogen


def get_device(prefer_mps=True, prefer_cuda=True):
    if prefer_mps and torch.backends.mps.is_available():
        return "mps"
    if prefer_cuda and torch.cuda.is_available():
        return "cuda"
    return "cpu"


def compute_model_density(model, device, r_max, grid_points, z0=0.0, n_value=1):
    x = torch.linspace(-r_max, r_max, grid_points, device=device)
    y = torch.linspace(-r_max, r_max, grid_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z0)

    radial = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2).clamp_min(1e-6).unsqueeze(-1)
    cos_theta = (zz / radial.squeeze(-1)).clamp(-1.0, 1.0)
    theta = torch.acos(cos_theta).unsqueeze(-1)
    phi = torch.atan2(yy, xx).unsqueeze(-1)
    n = torch.full_like(radial, float(n_value))

    with torch.no_grad():
        psi = model(radial, theta, phi, n)
        density = (psi ** 2).view(grid_points, grid_points).cpu()
    return density


def compute_analytic_density(r_max, grid_points, z0=0.0, device="cpu", n_value=1):
    x = torch.linspace(-r_max, r_max, grid_points, device=device)
    y = torch.linspace(-r_max, r_max, grid_points, device=device)
    xx, yy = torch.meshgrid(x, y, indexing="ij")
    zz = torch.full_like(xx, z0)
    radial = torch.sqrt(xx ** 2 + yy ** 2 + zz ** 2)

    if n_value == 1:
        # |psi_100|^2 = 1/pi * exp(-2r)
        density = (1.0 / torch.pi) * torch.exp(-2.0 * radial)
    elif n_value == 2:
        # |psi_200|^2 = 1/(32*pi) * (2 - r)^2 * exp(-r)
        density = (1.0 / (32.0 * torch.pi)) * (2.0 - radial) ** 2 * torch.exp(-radial)
    else:
        raise ValueError(f"Analytic density not implemented for n={n_value}")
    return density.cpu()


def resolve_output_path(base_output, n_value):
    base = pathlib.Path(base_output)
    base_str = str(base)
    if "{n}" in base_str:
        return pathlib.Path(base_str.format(n=n_value))
    if base.suffix:
        return base.with_name(f"{base.stem}_n{n_value}{base.suffix}")
    return base / f"n{n_value}.png"


def render_comparison(
    model_path="schrodingers_equation_hydrogen.pt",
    output_path="hydrogen_{n}s_compare.png",
    r_max=6.0,
    grid_points=220,
    z0=0.0,
    n_value=1,
    model=None,
    device=None,
):
    if device is None:
        device = get_device()
    if model is None:
        model = SchrodingerEquationHydrogen(L=1.0).to(device)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model.eval()

    model_density = compute_model_density(model, device, r_max, grid_points, z0, n_value)
    analytic_density = compute_analytic_density(r_max, grid_points, z0, device, n_value)
    error = (model_density - analytic_density).abs()
    print(f"Percentage max error for n={n_value}: {100.0 * error.max().item() / analytic_density.max().item():.4f}%")
    vmax = max(model_density.max().item(), analytic_density.max().item())

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    titles = ["Model |psi|^2", f"Analytic {n_value}s |psi|^2", "Absolute error"]
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

    fig.suptitle(f"Hydrogen {n_value}s density comparison at z = {z0:.2f} a.u.")
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    output_path = resolve_output_path(output_path, n_value)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220)
    plt.close(fig)
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Compare PINN hydrogen 1s with analytic density on a plane.")
    parser.add_argument("--model", default="schrodingers_equation_hydrogen.pt", help="Path to trained model weights")
    parser.add_argument("--out", default="hydrogen_{n}s_compare.png", help="Output image path or template (use {n} for state number)")
    parser.add_argument("--r_max", type=float, default=6.0, help="Spatial extent for the grid in atomic units")
    parser.add_argument("--grid", type=int, default=220, help="Grid resolution per axis")
    parser.add_argument("--z", type=float, default=0.0, help="z-plane to slice (a.u.)")
    parser.add_argument("--states", type=str, default="1,2", help="Comma-separated principal quantum numbers to visualise (e.g., 1,2)")
    args = parser.parse_args()
    device = get_device()
    model = SchrodingerEquationHydrogen(L=1.0).to(device)
    state_dict = torch.load(args.model, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    states = [int(s.strip()) for s in args.states.split(",") if s.strip()]
    if not states:
        raise ValueError("At least one state must be specified via --states")

    for state in states:
        output = render_comparison(
            model_path=args.model,
            output_path=args.out,
            r_max=args.r_max,
            grid_points=args.grid,
            z0=args.z,
            n_value=state,
            model=model,
            device=device,
        )
        print(f"Saved comparison image for n={state} to {output}")


if __name__ == "__main__":
    plt.switch_backend("Agg")
    main()
