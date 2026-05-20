import argparse
import json
import pathlib

import matplotlib.pyplot as plt
import numpy as np

from schrodingers_equation_h2 import (
    H2_BONDING_AUDIT_BATCH_SIZE,
    H2_BONDING_AUDIT_BOX_LENGTH,
    H2_ENERGY_NORM_ONLY_CHECKPOINT_PREFIX,
    H2_FULL_CHECKPOINT_PREFIX,
    H2_MODEL_DIR,
    H2_BONDING_AUDIT_REPEATS,
    _H2_BO_REFERENCE_ENERGIES,
    discover_h2_checkpoints,
    evaluate_h2_checkpoint,
    get_device,
    lookup_h2_reference_total_energy,
)

BOHR_TO_ANGSTROM = 0.529177210903
CM_PER_HARTREE = 219474.6313705
H1_MASS_AMU = 1.00782503223


def build_reference_curve():
    curve = sorted(
        (float(distance), float(energy))
        for distance, energy in _H2_BO_REFERENCE_ENERGIES.items()
    )
    return curve


def evaluate_checkpoints(checkpoints, device, box_length, batch_size, repeats):
    rows = []
    for nuclei_distance, checkpoint_path in checkpoints:
        predicted_energy, abs_error, quadrupole = evaluate_h2_checkpoint(
            checkpoint_path=checkpoint_path,
            nuclei_distance=nuclei_distance,
            device=device,
            box_length=box_length,
            batch_size=batch_size,
            n_repeats=repeats,
        )
        reference = lookup_h2_reference_total_energy(nuclei_distance)
        rows.append(
            {
                "R": float(nuclei_distance),
                "checkpoint": str(checkpoint_path),
                "predicted_quadrupole": float(quadrupole),
                "predicted_energy_hartree": float(predicted_energy),
                "reference_energy_hartree": float(reference["energy_hartree"]),
                "absolute_error_hartree": float(abs_error),
            }
        )
    return rows


def build_model_fit_curve(x_values, y_values, num_points):
    x = np.asarray(x_values, dtype=np.float64)
    y = np.asarray(y_values, dtype=np.float64)
    if x.ndim != 1 or y.ndim != 1 or x.size != y.size:
        raise ValueError("x_values and y_values must be 1D arrays of equal length.")
    if x.size < 2:
        return None, None, "none"

    dense_x = np.linspace(float(x[0]), float(x[-1]), int(max(2, num_points)))
    if x.size == 2:
        dense_y = np.interp(dense_x, x, y)
        return dense_x, dense_y, "linear"

    h = np.diff(x)
    if np.any(h <= 0.0):
        raise ValueError("Model R values must be strictly increasing for spline fitting.")

    a = y.copy()
    alpha = np.zeros_like(x)
    alpha[1:-1] = (
        3.0 / h[1:] * (a[2:] - a[1:-1])
        - 3.0 / h[:-1] * (a[1:-1] - a[:-2])
    )

    l = np.ones_like(x)
    mu = np.zeros_like(x)
    z = np.zeros_like(x)
    for idx in range(1, x.size - 1):
        l[idx] = 2.0 * (x[idx + 1] - x[idx - 1]) - h[idx - 1] * mu[idx - 1]
        mu[idx] = h[idx] / l[idx]
        z[idx] = (alpha[idx] - h[idx - 1] * z[idx - 1]) / l[idx]

    b = np.zeros(x.size - 1, dtype=np.float64)
    c = np.zeros_like(x)
    d = np.zeros(x.size - 1, dtype=np.float64)
    for idx in range(x.size - 2, -1, -1):
        c[idx] = z[idx] - mu[idx] * c[idx + 1]
        b[idx] = (a[idx + 1] - a[idx]) / h[idx] - h[idx] * (c[idx + 1] + 2.0 * c[idx]) / 3.0
        d[idx] = (c[idx + 1] - c[idx]) / (3.0 * h[idx])

    dense_y = np.empty_like(dense_x)
    interval_ids = np.searchsorted(x, dense_x, side="right") - 1
    interval_ids = np.clip(interval_ids, 0, x.size - 2)
    dx = dense_x - x[interval_ids]
    dense_y[:] = (
        a[interval_ids]
        + b[interval_ids] * dx
        + c[interval_ids] * dx ** 2
        + d[interval_ids] * dx ** 3
    )
    return dense_x, dense_y, "natural_cubic"


def save_summary(summary_path, rows, args, resolved_device, fit_kind):
    payload = {
        "models_dir": str(pathlib.Path(args.models_dir)),
        "model_family": "energy_norm_only" if args.energy_norm_only else "full_constraints",
        "checkpoint_prefix": H2_ENERGY_NORM_ONLY_CHECKPOINT_PREFIX
        if args.energy_norm_only
        else H2_FULL_CHECKPOINT_PREFIX,
        "eval_device_arg": args.eval_device,
        "resolved_device": resolved_device,
        "eval_box_length": float(args.eval_box_length),
        "eval_batch_size": int(args.eval_batch_size),
        "eval_repeats": int(args.eval_repeats),
        "fit_kind": fit_kind,
        "fit_points": int(args.fit_points),
        "results": rows,
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(payload, indent=2))


def format_level_number(value):
    text = f"{float(value):.15g}"
    return text.replace("e", "D").replace("E", "D")


def save_level16_input(level_input_path, rows, args, fit_kind):
    rows = sorted(rows, key=lambda row: float(row["R"]))
    if not rows:
        raise ValueError("Cannot export LEVEL input without PES rows.")

    level_rows = []
    for row in rows:
        energy_relative_to_dissociation = (
            float(row["predicted_energy_hartree"]) - float(args.level_dissociation_limit_hartree)
        )
        level_rows.append((float(row["R"]), energy_relative_to_dissociation))

    title = f"H2 X-state pointwise potential ({fit_kind} fit sample points)"
    lines = [
        " 1 1  1 1   0   1                  % IAN1 IMN1 IAN2 IMN2 CHARGE NUMPOT",
        # f" 'H ' {format_level_number(H1_MASS_AMU)}D0                  % NAME1 MASS1",
        # f" 'H ' {format_level_number(H1_MASS_AMU)}D0                  % NAME2 MASS2",
        f"'{title}'",
        (
            f" {format_level_number(args.level_rh_angstrom)}  "
            f"{format_level_number(args.level_rmin_angstrom)}  "
            f"{format_level_number(args.level_rmax_angstrom)}  "
            f"{format_level_number(args.level_eps_cm)}             % RH RMIN RMAX EPS"
        ),
        f" {len(level_rows)}  0  0  0.0D0                       % NTP LPPOT IOMEG VLIM",
        (
            f" 0  0  {int(args.level_tail_ilr)}  {int(args.level_tail_ncn)}  0.D0"
            "                    % NUSE IR2 ILR NCN CNN"
        ),
        (
            f" {format_level_number(BOHR_TO_ANGSTROM)}D0  "
            f"{format_level_number(CM_PER_HARTREE)}D0  0.0D0"
            "          % RFACT EFACT VSHIFT"
        ),
    ]
    lines.extend(
        f" {format_level_number(radius_bohr)}  {format_level_number(energy_hartree)}"
        for radius_bohr, energy_hartree in level_rows
    )
    lines.extend(
        [
            " -999  1  0  0  999  1  1  1           % NLEV1 AUTO1 LCDC LXPCT NJM JDJR IWR LPRWF",
            " 0 0                                % IV(1) IJ(1)",
        ]
    )

    level_input_path.parent.mkdir(parents=True, exist_ok=True)
    level_input_path.write_text("\n".join(lines) + "\n")


def plot_results(output_path, rows, title, fit_points):
    reference_curve = build_reference_curve()
    ref_r = [item[0] for item in reference_curve]
    ref_e = [item[1] for item in reference_curve]
    model_r = [row["R"] for row in rows]
    model_e = [row["predicted_energy_hartree"] for row in rows]
    model_ref = [row["reference_energy_hartree"] for row in rows]
    fit_x, fit_y, fit_kind = build_model_fit_curve(model_r, model_e, fit_points)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True, height_ratios=(3, 1.3))

    axes[0].plot(ref_r, ref_e, color="black", linewidth=2.0, label="Ground truth BO curve")
    axes[0].scatter(
        model_r,
        model_ref,
        color="tab:green",
        marker="x",
        s=60,
        label="Ground truth at trained R",
        zorder=3,
    )
    axes[0].scatter(
        model_r,
        model_e,
        color="tab:blue",
        marker="o",
        s=45,
        label="Model PES samples",
        zorder=4,
    )
    if fit_x is not None:
        fit_label = "Model fit (natural cubic spline)" if fit_kind == "natural_cubic" else "Model fit"
        axes[0].plot(
            fit_x,
            fit_y,
            color="tab:orange",
            linewidth=2.0,
            linestyle="--",
            label=fit_label,
            zorder=2,
        )
    axes[0].set_ylabel("Total Energy (Ha)")
    axes[0].set_title(title)
    axes[0].grid(alpha=0.3)
    axes[0].legend()

    error_values = [row["absolute_error_hartree"] for row in rows]
    axes[1].plot(model_r, error_values, color="tab:red", marker="o", linewidth=1.8, zorder=3)
    axes[1].axhline(0.0, color="black", linewidth=1.0, alpha=0.7)
    axes[1].set_xlabel("Internuclear Distance R (Bohr)")
    axes[1].set_ylabel("|Error| (Ha)")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return fit_kind


def build_parser():
    parser = argparse.ArgumentParser(
        description="Plot the H2 potential energy surface for all trained checkpoints against the BO reference curve."
    )
    parser.add_argument(
        "--models-dir",
        type=pathlib.Path,
        default=H2_MODEL_DIR,
        help="Directory containing R-tagged H2 checkpoints.",
    )
    parser.add_argument(
        "--energy-norm-only",
        action="store_true",
        help=(
            "Evaluate checkpoints trained with only Rayleigh energy minimization and normalization. "
            "Defaults outputs to energy_norm_only_h2_pes_compare.* so they do not overwrite full-constraint PES files."
        ),
    )
    parser.add_argument(
        "--eval-device",
        type=str,
        default="auto",
        help="Evaluation device: auto, cpu, cuda, or mps. Matches the post-training audit device role.",
    )
    parser.add_argument(
        "--eval-box-length",
        type=float,
        default=H2_BONDING_AUDIT_BOX_LENGTH,
        help="QMC box length used by the post-training repeated-Sobol audit.",
    )
    parser.add_argument(
        "--eval-batch-size",
        type=int,
        default=H2_BONDING_AUDIT_BATCH_SIZE,
        help="Sobol sample count per repeat and checkpoint. Matches training audit defaults.",
    )
    parser.add_argument(
        "--eval-repeats",
        type=int,
        default=H2_BONDING_AUDIT_REPEATS,
        help="Independent repeated Sobol evaluations per checkpoint. Matches training audit defaults.",
    )
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=pathlib.Path("assets/h2_pes_compare.png"),
        help="Path to save the PES comparison figure.",
    )
    parser.add_argument(
        "--summary-json",
        type=pathlib.Path,
        default=None,
        help="Optional JSON path for the evaluated checkpoint summary. Defaults to output stem + .json.",
    )
    parser.add_argument(
        "--title",
        type=str,
        default="H2 Potential Energy Surface: Trained Models vs Ground Truth",
        help="Figure title.",
    )
    parser.add_argument(
        "--fit-points",
        type=int,
        default=400,
        help="Number of dense points used to draw the fitted model curve.",
    )
    parser.add_argument(
        "--level-input",
        type=pathlib.Path,
        default=None,
        help="Optional path for a LEVEL16 pointwise-potential input file. Defaults to output stem + _level16.in.",
    )
    parser.add_argument(
        "--level-rh-angstrom",
        type=float,
        default=0.0005,
        help="LEVEL integration grid spacing RH in Angstrom.",
    )
    parser.add_argument(
        "--level-rmin-angstrom",
        type=float,
        default=0.1,
        help="LEVEL integration lower bound RMIN in Angstrom.",
    )
    parser.add_argument(
        "--level-rmax-angstrom",
        type=float,
        default=40.0,
        help="LEVEL integration upper bound RMAX in Angstrom.",
    )
    parser.add_argument(
        "--level-eps-cm",
        type=float,
        default=1.0e-8,
        help="LEVEL eigenvalue convergence tolerance EPS in cm^-1.",
    )
    parser.add_argument(
        "--level-tail-ilr",
        type=int,
        default=2,
        help="LEVEL ILR tail extrapolation flag for the exported pointwise potential.",
    )
    parser.add_argument(
        "--level-tail-ncn",
        type=int,
        default=6,
        help="LEVEL NCN leading inverse-power exponent for the exported pointwise potential tail.",
    )
    parser.add_argument(
        "--level-dissociation-limit-hartree",
        type=float,
        default=-1.0,
        help="Dissociation limit used to shift exported PES samples to LEVEL's VLIM=0 convention.",
    )
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    checkpoints = discover_h2_checkpoints(
        args.models_dir, energy_norm_only=args.energy_norm_only
    )
    if not checkpoints:
        model_kind = "energy/norm-only" if args.energy_norm_only else "full-constraint"
        parser.error(f"No {model_kind} R-tagged H2 checkpoints were found in {args.models_dir}.")

    device = get_device(args.eval_device)
    rows = evaluate_checkpoints(
        checkpoints=checkpoints,
        device=device,
        box_length=args.eval_box_length,
        batch_size=args.eval_batch_size,
        repeats=args.eval_repeats,
    )

    output_path = pathlib.Path(args.output)
    if args.energy_norm_only and output_path == pathlib.Path("assets/h2_pes_compare.png"):
        output_path = pathlib.Path("assets/energy_norm_only_h2_pes_compare.png")
    title = args.title
    if (
        args.energy_norm_only
        and title == "H2 Potential Energy Surface: Trained Models vs Ground Truth"
    ):
        title = "H2 Potential Energy Surface: Energy/Norm-Only Models vs Ground Truth"
    summary_path = (
        pathlib.Path(args.summary_json)
        if args.summary_json is not None
        else output_path.with_suffix(".json")
    )
    level_input_path = (
        pathlib.Path(args.level_input)
        if args.level_input is not None
        else output_path.with_name(f"{output_path.stem}_level16.in")
    )

    fit_kind = plot_results(
        output_path=output_path,
        rows=rows,
        title=title,
        fit_points=args.fit_points,
    )
    save_summary(
        summary_path=summary_path,
        rows=rows,
        args=args,
        resolved_device=device,
        fit_kind=fit_kind,
    )
    save_level16_input(
        level_input_path=level_input_path,
        rows=rows,
        args=args,
        fit_kind=fit_kind,
    )

    print(f"Saved PES plot to {output_path}")
    print(f"Saved evaluation summary to {summary_path}")
    print(f"Saved LEVEL16 pointwise input to {level_input_path}")
    print(f"Model fit: {fit_kind}")
    num_unbound_points = sum(
        1
        for row in rows
        if row["predicted_energy_hartree"] > args.level_dissociation_limit_hartree
    )
    if num_unbound_points:
        print(
            "Warning: "
            f"{num_unbound_points} exported PES samples lie above the chosen dissociation limit "
            f"({args.level_dissociation_limit_hartree:.6f} Ha)."
        )
    for row in rows:
        print(
            f"R={row['R']:.6g} | E_model={row['predicted_energy_hartree']:.9f} Ha | "
            f"E_ref={row['reference_energy_hartree']:.9f} Ha | "
            f"|err|={row['absolute_error_hartree']:.3e} Ha"
        )


if __name__ == "__main__":
    main()
