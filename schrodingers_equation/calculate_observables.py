from collections import defaultdict
import argparse
import json
from pathlib import Path
import re

import matplotlib.pyplot as plt

import numpy as np

from scipy.interpolate import PchipInterpolator
from scipy.linalg import eigh_tridiagonal
from scipy.signal import find_peaks

from sympy.physics.wigner import wigner_3j
from sympy import N

ASSETS_DIR = Path(__file__).resolve().parent / "assets"
DEFAULT_SUMMARY_PATH = ASSETS_DIR / "h2_pes_compare.json"
DEFAULT_LEVEL16_OUTPUT_PATH = ASSETS_DIR / "h2_pes_compare_level16.out"
ENERGY_NORM_ONLY_SUMMARY_PATH = ASSETS_DIR / "energy_norm_only_h2_pes_compare.json"
ENERGY_NORM_ONLY_LEVEL16_OUTPUT_PATH = ASSETS_DIR / "energy_norm_only_h2_pes_compare_level16.out"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compute H2 bound/continuum observables from a saved PES summary and LEVEL16 output."
    )
    parser.add_argument(
        "--energy-norm-only-pes",
        action="store_true",
        help=(
            "Use the PES generated from energy/norm-only H2 models instead of the default "
            "full-constraint PES."
        ),
    )
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="Override the PES summary JSON path.",
    )
    parser.add_argument(
        "--level16-output",
        type=Path,
        default=None,
        help="Override the LEVEL16 output path used for bound wavefunctions.",
    )
    return parser.parse_args()


ARGS = parse_args()
SUMMARY_PATH = (
    ARGS.summary_json
    if ARGS.summary_json is not None
    else ENERGY_NORM_ONLY_SUMMARY_PATH
    if ARGS.energy_norm_only_pes
    else DEFAULT_SUMMARY_PATH
)
LEVEL16_OUTPUT_PATH = (
    ARGS.level16_output
    if ARGS.level16_output is not None
    else ENERGY_NORM_ONLY_LEVEL16_OUTPUT_PATH
    if ARGS.energy_norm_only_pes
    else DEFAULT_LEVEL16_OUTPUT_PATH
)

with SUMMARY_PATH.open("r") as f:
    content = json.load(f)

x = [i["R"] for i in content["results"]]
y_energy = [i["predicted_energy_hartree"] for i in content["results"]]
y_quadrupole = [i["predicted_quadrupole"] for i in content["results"]]

# Fit potential energy surface
V = PchipInterpolator(x, y_energy)
PES_RIGHT_CLAMP_R = 6.0
PES_RIGHT_CLAMP_VALUE = float(V(PES_RIGHT_CLAMP_R))
PES_DISSOCIATION_LIMIT = -1.0
PES_TAIL_DECAY_LENGTH = 2.0

# Hydrogen mass comes from https://physics.nist.gov/cgi-bin/Compositions/stand_alone.pl?ele=H
# I had to convert from AMU to electron masses
reduced_mass = 1837.152708434921 / 2.0

GRID_R_MIN = 0.5
GRID_R_MAX = 40.0
GRID_N = 20000
MAX_J = 5
NUM_BOUND_LEVELS_TO_REPORT = 5
NUM_CONTINUUM_STATES_TO_KEEP = 18
INITIAL_EIGENSTATE_COUNT = 500
DIAGNOSTIC_CONTINUUM_STATES = 3
CM_PER_HARTREE = 219474.6313705
EV_PER_HARTREE = 27.211
BOHR_TO_ANGSTROM = 0.529177210903
HARTREE_TO_JOULE = 4.35974e-18
ELEMENTARY_CHARGE = 1.602176634e-19
BOHR_RADIUS = 5.29177210544e-11
PLANCK_CONSTANT = 6.62607015e-34
AVOGADRO_NUMBER = 6.02214076e23
PERMITTIVITY_VACUUM = 8.85418782e-12
SPEED_OF_LIGHT = 299792458.0
BOLTZMANN_CONSTANT = 3.166829681e-6  # Boltzmann constant in Hartree/K
FINE_STRUCTURE = 7.2973525693e-3
RYDBERG_CONSTANT = 10973731.568160
IDEAL_GAS_CONSTANT = AVOGADRO_NUMBER * BOLTZMANN_CONSTANT * HARTREE_TO_JOULE
MOLAR_MASS_H2 = 2.01588  # g/mol
continuum_E_min = 0.001
continuum_E_max = 0.01

LEVEL16_HEADER_RE = re.compile(
    r"Solution of radial Schr\. equation for\s+"
    r"E\(v=\s*(\d+),J=\s*(\d+)\)\s*=\s*"
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?)"
)
LEVEL16_FLOAT_RE = re.compile(
    r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[DEde][+-]?\d+)?"
)


def evaluate_pes_with_asymptotic_tail(
    interpolator,
    r_values,
    match_radius,
    match_value,
    asymptote,
    decay_length,
):
    r_values = np.asarray(r_values)
    values = interpolator(np.minimum(r_values, match_radius))
    tail_mask = r_values > match_radius
    tail_offset = match_value - asymptote
    tail_values = asymptote + tail_offset * np.exp(-(r_values - match_radius) / decay_length)
    if np.isscalar(values):
        return float(tail_values) if float(r_values) > match_radius else float(values)
    values = np.asarray(values, dtype=np.float64)
    values[tail_mask] = tail_values[tail_mask]
    return values


def evaluate_curve_with_right_clamp(interpolator, r_values, clamp_radius):
    return interpolator(np.minimum(np.asarray(r_values), clamp_radius))


def angular_factor(Ji, Jf):
    val = wigner_3j(Jf, 2, Ji, 0, 0, 0)
    val = float(N(val))
    return (2 * Jf + 1) * (val ** 2)


def parse_fortran_float(text):
    return float(text.replace("D", "E").replace("d", "e"))


def collapse_duplicate_radii(r_values, wavefunction_values):
    unique_r, inverse = np.unique(r_values, return_inverse=True)
    wf_sum = np.zeros_like(unique_r)
    counts = np.zeros_like(unique_r)
    np.add.at(wf_sum, inverse, wavefunction_values)
    np.add.at(counts, inverse, 1.0)
    return unique_r, wf_sum / counts


def parse_level16_wavefunction_sections(level_output_path):
    lines = level_output_path.read_text(errors="ignore").splitlines()
    sections = []
    line_idx = 0

    while line_idx < len(lines):
        match = LEVEL16_HEADER_RE.search(lines[line_idx])
        if match is None:
            line_idx += 1
            continue

        vibrational_index = int(match.group(1))
        rotational_index = int(match.group(2))
        energy_cm = parse_fortran_float(match.group(3))
        line_idx += 1

        radii_angstrom = []
        wavefunction_values = []

        while line_idx < len(lines):
            stripped = lines[line_idx].strip()
            if stripped.startswith("Solution of radial Schr. equation for") or stripped.startswith("E(v="):
                break
            if not stripped or stripped.startswith("R(I)") or set(stripped) == {"-"}:
                line_idx += 1
                continue

            numeric_fields = LEVEL16_FLOAT_RE.findall(lines[line_idx])
            if len(numeric_fields) % 2 != 0:
                raise ValueError(
                    f"Unexpected LEVEL16 wavefunction row format: {lines[line_idx]!r}"
                )

            for radius_text, value_text in zip(numeric_fields[0::2], numeric_fields[1::2]):
                radii_angstrom.append(parse_fortran_float(radius_text))
                wavefunction_values.append(parse_fortran_float(value_text))
            line_idx += 1

        if not radii_angstrom:
            raise ValueError(
                f"Found LEVEL16 section for v={vibrational_index}, J={rotational_index} "
                "without any wavefunction samples."
            )

        sections.append(
            {
                "v": vibrational_index,
                "J": rotational_index,
                "energy_cm": energy_cm,
                "r_angstrom": np.asarray(radii_angstrom, dtype=np.float64),
                "wavefunction_angstrom": np.asarray(wavefunction_values, dtype=np.float64),
            }
        )

    if not sections:
        raise ValueError(f"No LEVEL16 wavefunction sections were parsed from {level_output_path}.")

    return sections


def resample_level16_wavefunction_to_grid(section, r_grid_bohr):
    r_angstrom, wavefunction_angstrom = collapse_duplicate_radii(
        section["r_angstrom"],
        section["wavefunction_angstrom"],
    )
    r_bohr = r_angstrom / BOHR_TO_ANGSTROM
    wavefunction_bohr = np.sqrt(BOHR_TO_ANGSTROM) * wavefunction_angstrom
    resampled_wavefunction = np.interp(r_grid_bohr, r_bohr, wavefunction_bohr, left=0.0, right=0.0)
    norm = np.sqrt(np.trapz(resampled_wavefunction**2, r_grid_bohr))
    if norm <= 1.0e-12:
        raise ValueError(
            f"LEVEL16 wavefunction for v={section['v']}, J={section['J']} vanished after resampling."
        )
    return resampled_wavefunction / norm


def load_level16_bound_spectrum(level_output_path, r_grid_bohr, max_j, min_bound_levels):
    grouped_sections = defaultdict(list)
    for section in parse_level16_wavefunction_sections(level_output_path):
        if section["J"] >= max_j or section["energy_cm"] >= 0.0:
            continue
        grouped_sections[section["J"]].append(section)

    bound_energies = []
    bound_wavefunctions = []

    for J in range(max_j):
        sections_for_j = sorted(grouped_sections.get(J, []), key=lambda section: section["v"])
        if len(sections_for_j) < min_bound_levels:
            raise RuntimeError(
                f"LEVEL16 output only provided {len(sections_for_j)} negative-energy states for J={J}; "
                f"need at least {min_bound_levels}."
            )

        j_energies = np.asarray(
            [section["energy_cm"] / CM_PER_HARTREE for section in sections_for_j],
            dtype=np.float64,
        )
        j_wavefunctions = np.column_stack(
            [resample_level16_wavefunction_to_grid(section, r_grid_bohr) for section in sections_for_j]
        )

        bound_energies.append(j_energies)
        bound_wavefunctions.append(j_wavefunctions)

    return bound_energies, bound_wavefunctions


# def solve_low_energy_box_spectrum(
#     r_grid,
#     potential_relative_to_threshold,
#     max_J,
#     min_bound_states,
#     min_continuum_states,
#     initial_state_count,
# ):
#     dR = r_grid[1] - r_grid[0]
#     energies = []
#     wavefunctions = []

#     for J in range(max_J+1):
#         V_eff = potential_relative_to_threshold + J * (J + 1) / (2 * reduced_mass * r_grid**2)
#         diag = np.full(r_grid.size, 1.0 / (reduced_mass * dR**2)) + V_eff
#         offdiag = np.full(r_grid.size - 1, -0.5 / (reduced_mass * dR**2))

#         state_count = min(initial_state_count, r_grid.size)
#         while True:
#             j_energies, j_wavefunctions = eigh_tridiagonal(
#                 diag,
#                 offdiag,
#                 select="v",
#                 select_range=(-np.inf, -1e-12),  # only E < 0 bound states
#             )
#             num_bound = np.count_nonzero(j_energies < 0.0)
#             if num_bound >= min_bound_states:
#                 break
#             if state_count == r_grid.size:
#                 raise RuntimeError(
#                     f"Could not find enough bound/continuum states for J={J} on the shared grid."
#                 )
#             state_count = min(2 * state_count, r_grid.size)

#         norms = np.sqrt(np.trapz(j_wavefunctions**2, r_grid, axis=0))
#         j_wavefunctions = j_wavefunctions / norms
#         energies.append(j_energies)
#         wavefunctions.append(j_wavefunctions)

#     return energies, wavefunctions


# def split_spectrum_by_sign(energies, wavefunctions):
#     bound_energies = []
#     bound_wavefunctions = []
#     continuum_energies = []
#     continuum_wavefunctions = []

#     for j_energies, j_wavefunctions in zip(energies, wavefunctions):
#         bound_mask = j_energies < 0.0
#         continuum_mask = (j_energies > continuum_E_min) & (j_energies < continuum_E_max)
#         bound_energies.append(j_energies[bound_mask])
#         bound_wavefunctions.append(j_wavefunctions[:, bound_mask])
#         continuum_energies.append(j_energies[continuum_mask])
#         continuum_wavefunctions.append(j_wavefunctions[:, continuum_mask])

#     return bound_energies, bound_wavefunctions, continuum_energies, continuum_wavefunctions

def solve_scattering_state_numerov(R, V_eff, E, reduced_mass, J, match_fraction=0.75):
    h = R[1] - R[0]

    # Rewrite the Schrodinger equation:
    # chi'' = 2 mu (V_eff - E) chi
    f = 2.0 * reduced_mass * (V_eff - E)

    chi = np.zeros_like(R)

    # Regular radial behavior near small R: chi ~ R^(J+1)
    chi[0] = R[0] ** (J + 1)
    chi[1] = R[1] ** (J + 1)

    for i in range(1, len(R) - 1):
        chi[i + 1] = (
            2.0 * chi[i] * (1.0 + 5.0 * h**2 * f[i] / 12.0)
            - chi[i - 1] * (1.0 - h**2 * f[i - 1] / 12.0)
        ) / (1.0 - h**2 * f[i + 1] / 12.0)

    k = np.sqrt(2.0 * reduced_mass * E)

    start = int(match_fraction * len(R))
    R_tail = R[start:]
    chi_tail = chi[start:]

    phase = k * R_tail - J * np.pi / 2.0

    basis = np.column_stack([
        np.sin(phase),
        np.cos(phase),
    ])

    C, D = np.linalg.lstsq(basis, chi_tail, rcond=None)[0]
    amplitude = np.sqrt(C**2 + D**2)

    if not np.isfinite(amplitude) or amplitude <= 1e-30:
        raise RuntimeError(f"Bad scattering normalization at E={E}, J={J}")

    # Energy-normalized asymptotic amplitude in atomic units
    target_amplitude = np.sqrt(2.0 * reduced_mass / (np.pi * k))

    return chi * (target_amplitude / amplitude)

def save_current_figure(path):
    fig = plt.gcf()
    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight", pad_inches=0.25)


R = np.linspace(GRID_R_MIN, GRID_R_MAX, GRID_N)
V_abs = evaluate_pes_with_asymptotic_tail(
    V,
    R,
    PES_RIGHT_CLAMP_R,
    PES_RIGHT_CLAMP_VALUE,
    PES_DISSOCIATION_LIMIT,
    PES_TAIL_DECAY_LENGTH,
)
V_rel = V_abs - PES_DISSOCIATION_LIMIT

quadrupole_interpolator = PchipInterpolator(x, y_quadrupole)
quadrupole_vals = evaluate_curve_with_right_clamp(quadrupole_interpolator, R, max(x))
D_vals = -0.5 * quadrupole_vals

bound_energies, bound_wavefuncs = load_level16_bound_spectrum(
    LEVEL16_OUTPUT_PATH,
    R,
    MAX_J,
    NUM_BOUND_LEVELS_TO_REPORT,
)

# box_energies, box_wavefunctions = solve_low_energy_box_spectrum(
#     R,
#     V_rel,
#     MAX_J,
#     0,
#     NUM_CONTINUUM_STATES_TO_KEEP,
#     INITIAL_EIGENSTATE_COUNT,
# )

# _, _, cont_energies, cont_wavefuncs = split_spectrum_by_sign(
#     box_energies,
#     box_wavefunctions,
# )

E_cont_grid = np.linspace(continuum_E_min, continuum_E_max, NUM_CONTINUUM_STATES_TO_KEEP)



for J in range(MAX_J):
    print(
        f"J = {J}: {len(bound_energies[J])} LEVEL16 bound states, "
        f"{NUM_CONTINUUM_STATES_TO_KEEP} positive-energy box states"
    )
print()

reported_bound_energies = np.stack(
    [j_energies[:NUM_BOUND_LEVELS_TO_REPORT] for j_energies in bound_energies],
    axis=0,
)

# Zero point energy
zpe = reported_bound_energies[0, 0] - V_rel.min()
print(f"Zero point energy: {CM_PER_HARTREE * zpe} cm^-1")
print("Reference zero point energy: 2080.6 cm^-1")  # From https://cccbdb.nist.gov/exp2x.asp?casno=1333740
print(f"Absolute error: {abs(2080.6 - CM_PER_HARTREE * zpe)} cm^-1")
print()

# Vibrational frequencies
frequencies = np.diff(reported_bound_energies[0], n=1)
print("Predicted vibrational spacings:")
for i, freq in enumerate(frequencies.tolist()):
    print(f"Spacing v = {i} -> {i + 1}: {CM_PER_HARTREE * freq} cm^-1")
    if i == 0:
        print("Reference: 4161 cm^-1")
        print(f"Absolute error: {abs(4161 - CM_PER_HARTREE * freq)} cm^-1")
print()

# omega_e_x_e
omega_e_x_e = (frequencies[0] - frequencies[1]) / 2.0
print(f"Predicted omega_e_x_e: {CM_PER_HARTREE * omega_e_x_e} cm^-1")
print("Reference omega_e_x_e: 121.336 cm^-1")
print(f"Absolute error: {abs(121.336 - CM_PER_HARTREE * omega_e_x_e)} cm^-1")
print()

# omega_e
omega_e = frequencies[0] + 2 * omega_e_x_e
print(f"Predicted omega_e: {CM_PER_HARTREE * omega_e} cm^-1")
print("Reference omega_e: 4401.213 cm^-1")
print(f"Absolute error: {abs(4401.213 - CM_PER_HARTREE * omega_e)} cm^-1")
print()

# Vibrational ladder between rotational levels
print("Vibrational ladder between rotational levels")
for v in range(min(3, NUM_BOUND_LEVELS_TO_REPORT)):
    print(f"v = {v}")
    for J in range(1, MAX_J):
        dE = bound_energies[J][v] - bound_energies[0][v]
        print(f"J = {J}, dE = {CM_PER_HARTREE * dE}")
print()

# Rotational constants
for i in range(NUM_BOUND_LEVELS_TO_REPORT):
    E0 = bound_energies[0][i]
    E1 = bound_energies[1][i]
    B_v = (E1 - E0) / 2.0
    print(f"B_v (from spectrum) v={i}: {CM_PER_HARTREE * B_v} cm^-1")
print()

# Equilibrium bond length
index = np.argmin(V_abs)
bond_length = R[index]
print(f"Equilibrium bond length: {bond_length / 1.89} Å")
print("Reference bond length: 0.7414Å")  # From https://cccbdb.nist.gov/exp2x.asp?casno=1333740
print(f"Absolute error: {abs(0.7414 - bond_length / 1.89)} Å")

# Dissociation energy
dissociation_energy = (0.0 - V_rel.min()) * EV_PER_HARTREE
print(f"Dissociation energy: {dissociation_energy} eV")
print("Reference dissociation energy: 4.74772565275935 eV")
print(f"Absolute error: {abs(dissociation_energy - 4.74772565275935)} eV")
print()

# Positive-energy boxed pseudostate diagnostics on the shared grid.

plt.figure(figsize=(10, 6))
plt.xlabel("Internuclear distance R (Bohr)")
plt.ylabel("Wavefunction amplitude (arb. units)")
# for idx, E in enumerate(cont_energies[0][:DIAGNOSTIC_CONTINUUM_STATES]):
#     chi = cont_wavefuncs[0][:, idx]
for idx, E in enumerate(E_cont_grid):
    chi = solve_scattering_state_numerov(
        R,
        V_rel,
        E,
        reduced_mass,
        0,
    )
    mask = R > 15.0
    R_tail = R[mask]
    chi_tail = chi[mask]
    k = np.sqrt(2 * reduced_mass * E)
    lam = 2 * np.pi / k

    print(f"Continuum pseudostate {idx} (J=0): E = {E}")
    print("expected lambda:", lam)

    peaks, _ = find_peaks(chi_tail)
    peak_positions = R_tail[peaks]
    sign_changes = np.where(np.signbit(chi_tail[1:]) != np.signbit(chi_tail[:-1]))[0]
    zero_crossings = R_tail[sign_changes]

    print("peak positions:", peak_positions[:10])
    print("zero crossings:", zero_crossings[:10])
    if len(peak_positions) > 1:
        spacings = np.diff(peak_positions)
        print("mean peak spacing:", np.mean(spacings))
    print()

    plt.plot(R, chi, label=f"n={idx}, E={E:.6f}")

plt.legend()
save_current_figure(ASSETS_DIR / "continuum_pseudostate_diagnostics.png")

transitions = []
for Ji in range(MAX_J):

    # Compute dE using central differences
    # delta_E = np.zeros_like(E_list)

    # delta_E[1:-1] = 0.5 * (E_list[2:] - E_list[:-2])
    # delta_E[0] = E_list[1] - E_list[0]
    # delta_E[-1] = E_list[-1] - E_list[-2]

    V_eff_i = V_rel + Ji * (Ji + 1) / (2 * reduced_mass * R**2)

    for n, E_cont in enumerate(E_cont_grid):
        chi_cont = solve_scattering_state_numerov(
            R,
            V_eff_i,
            E_cont,
            reduced_mass,
            Ji,
        )
        for Jf in range(MAX_J):
            S_J = angular_factor(Ji, Jf)
            if S_J == 0.0:
                continue
                
            for v, E_bound in enumerate(bound_energies[Jf]):
                chi_bound = bound_wavefuncs[Jf][:, v]
                # dE = delta_E[n]
                I = np.trapz(chi_bound * D_vals * chi_cont, R)
                M2 = S_J * np.abs(I) ** 2
                omega = E_cont - E_bound
                if omega <= 0:
                    # Skip invalid transitions that would require absorption of a photon
                    continue
                # nu = (omega * HARTREE_TO_JOULE) / PLANCK_CONSTANT # Convert hartree energy to frequency in Hz
                # M2_SI = M2 * (ELEMENTARY_CHARGE * BOHR_RADIUS**2)**2
                # prefactor = (8*np.pi**5) / (5 * PERMITTIVITY_VACUUM * PLANCK_CONSTANT * SPEED_OF_LIGHT**5)
                prefactor = (
                    4.0 * np.pi * RYDBERG_CONSTANT * SPEED_OF_LIGHT * FINE_STRUCTURE**5 / 15.0
                )
                A_density = prefactor * omega**5 * M2# * (nu**5) * M2_SI
                transitions.append(
                    {
                        "Ji": Ji,
                        "Jf": Jf,
                        "v": v,
                        "n": n,
                        "E_cont": E_cont,
                        "E_bound": E_bound,
                        "omega": omega,
                        "M2": M2,
                        "A_density": A_density,
                    }
                )
print(f"Total number of transitions: {len(transitions)}. Strongest 10:")
transitions.sort(key=lambda x: x["A_density"], reverse=True)
for t in transitions[:10]:
    print(
        f"Ji={t['Ji']} -> Jf={t['Jf']}, v={t['v']}, "
        f"E_cont={t['E_cont']:.6f}, E_bound={t['E_bound']:.6f}, "
        f"M2={t['M2']:.3e}, A_density={t['A_density']:.3e}"
    )
omegas = [t["omega"]*CM_PER_HARTREE for t in transitions]
delta_E_grid = np.gradient(E_cont_grid)
intensities = [t["A_density"] for t in transitions]
intensities_binned = [t["A_density"] * delta_E_grid[t["n"]] for t in transitions]
color = [t["Ji"] for t in transitions]

population = defaultdict(float)

for t in transitions:
    key = (t["v"], t["Jf"])
    dE = delta_E_grid[t["n"]]  # store n in transition dict
    population[key] += t["A_density"] * dE

# for t in transitions:
#     key = (t["v"], t["Jf"])
#     population[key] += t["A_density"]

sorted_pop = sorted(population.items(), key=lambda x: x[1], reverse=True)
print("Most populated bound states from scattering continuum transitions:")
for (v, Jf), val in sorted_pop[:10]:
    print(f"v={v}, J={Jf}, population={val}")

# Visualize the population distribution over bound states
max_v = max(v for v, _ in population.keys())
max_J = max(J for _, J in population.keys())

grid = np.zeros((max_v+1, max_J+1))

for (v, J), val in population.items():
    grid[v, J] = val

plt.figure(figsize=(10, 7))
grid_plot = grid / np.max(grid) if np.max(grid) > 0 else grid
plt.imshow(grid_plot, origin='lower', aspect='auto')
plt.colorbar(label="Normalized integrated continuum contribution")
plt.xlabel("Rotational quantum number J")
plt.ylabel("Vibrational quantum number v")
plt.title("Population distribution over bound states from scattering continuum transitions")
save_current_figure(ASSETS_DIR / "population_distribution.png")

# Scatter plot of transition energies vs. relative intensities, colored by initial J
plt.figure(figsize=(10, 7))
sc = plt.scatter(omegas, intensities, c=color)
cbar = plt.colorbar(sc)
cbar.set_label("Initial J_i")
plt.xlabel("Transition energy (cm^-1)")
plt.yscale("log")
plt.ylabel("Scattering-normalized E2 rate density")
plt.title("Quadrupole continuum -> bound transition strength from scattering states")
save_current_figure(ASSETS_DIR / "transition_contributions_scatter.png")

plt.figure(figsize=(10, 6))
plt.hist(omegas, weights=intensities_binned, bins=100)
plt.yscale("log")
plt.xlabel("Transition energy (cm^-1)")
plt.ylabel("Integrated continuum contribution")
plt.title("Binned quadrupole continuum-to-bound transition distribution")
save_current_figure(ASSETS_DIR / "transition_contributions_histogram.png")

# Compute partition function and thermochemical values at a given temperature
T = 300.0  # Temperature in K

Z = 0.0
E2_avg_numerator = 0.0
E_avg_numerator = 0.0

for J in range(len(bound_energies)):
    for v in range(len(bound_energies[J])):
        E = bound_energies[J][v] - bound_energies[0][0]  # Relative to lowest rovibrational level
        g_J = 2 * J + 1  # Degeneracy of rotational level
        # Para/ortho ratio for H2: even J (para) has weight 1, odd J (ortho) has weight 3
        if J % 2 == 0:
            g_J *= 1.0
        else:
            g_J *= 3.0
        w = g_J * np.exp(-E / (BOLTZMANN_CONSTANT * T))
        Z += w
        E_avg_numerator += E * w
        E2_avg_numerator += E**2 * w

E_avg = E_avg_numerator / Z
E2_avg = E2_avg_numerator / Z

# Compute heat capacity at constant volume
C_V = HARTREE_TO_JOULE * AVOGADRO_NUMBER * (E2_avg - E_avg**2) / (BOLTZMANN_CONSTANT * T**2 * MOLAR_MASS_H2) + 1.5 * IDEAL_GAS_CONSTANT / MOLAR_MASS_H2 # J/gK
C_P = C_V + IDEAL_GAS_CONSTANT / MOLAR_MASS_H2  # J/gK
print()
print("Thermochemical properties:")
print(f"Partition function at {T} K: {Z}")
print(f"Heat capacity at constant volume at {T} K: {C_V:.3f} J/gK")
print(f"Reference heat capacity at constant volume at {T} K: 10.16 J/gK")
print(f"Absolute error: {abs(C_V - 10.16):.3f} J/gK")
print()
print(f"Heat capacity at constant pressure at {T} K: {C_P:.3f} J/gK")
print(f"Reference heat capacity at constant pressure at {T} K: 14.304 J/gK")
print(f"Absolute error: {abs(C_P - 14.304):.3f} J/gK")
plt.show()
